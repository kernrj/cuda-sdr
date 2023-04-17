/*
 * Copyright 2023 Rick Kern <kernrj@gmail.com>
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "RfToPcmAudioFactory.h"

#include <ParseJson.h>
#include <util/util.h>

#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

using namespace std;
using namespace nlohmann;

static size_t bellangerLowPassTapCount(
    float sampleRateHz,
    float transitionWidthHz,
    float passBandRipple,
    float dbAttenuation) {
  const float sigma1 = pow(10.0f, passBandRipple / 10.0f);
  const float sigma2 = pow(10.0f, dbAttenuation / 10.0f);
  const float logVal = log10(1.0f / (10.0f * sigma1 * sigma2));
  const float end = sampleRateHz / transitionWidthHz;
  auto tapCount = lrint(ceilf(2.0f / 3.0f * logVal * end));

  return tapCount;
}

static size_t fredHarrisLowPassTapCount(float dbAttenuation, float transitionWidthHz, float sampleRateHz) {
  const float normalizedTransitionWidth = transitionWidthHz / sampleRateHz;
  return lrint(ceilf(-dbAttenuation / (22.0f * normalizedTransitionWidth)));
}

static Status createLowPassTaps(
    vector<float>* tapsOut,
    float sampleFrequency,
    float cutoffFrequency,
    float transitionWidth,
    float dbAttenuation) noexcept {
  try {
    const size_t bellangerTapCount = bellangerLowPassTapCount(sampleFrequency, transitionWidth, 0.01f, dbAttenuation);
    const size_t fredHarrisTapCount = fredHarrisLowPassTapCount(dbAttenuation, transitionWidth, sampleFrequency);

    gslogd(
        "Sample rate [%f] cutoff [%f] transition [%f] Bellanger count [%zu] Fred Harris count [%zu]",
        sampleFrequency,
        cutoffFrequency,
        transitionWidth,
        bellangerTapCount,
        fredHarrisTapCount);

    vector<size_t> tryTapLengths = {
        fredHarrisTapCount,
        bellangerTapCount,
        fredHarrisTapCount / 2,
        bellangerTapCount / 2};

    auto maxSizeIt = max_element(begin(tryTapLengths), end(tryTapLengths));
    GS_REQUIRE_OR_RET(
        maxSizeIt != tryTapLengths.end(),
        "Empty length vector computing low-pass taps",
        Status_InvalidArgument);
    const size_t maxSize = *maxSizeIt;

    vector<double> taps(maxSize);
    cout << "Creating low-pass taps with sample frequency [" << sampleFrequency << "] cutoff [" << cutoffFrequency
         << "] transition width [" << transitionWidth << "] attenuation [" << dbAttenuation << "] taps size ["
         << taps.size() << "]" << endl;

    RemezStatus status = RemezSuccess;

    for (size_t tapLength : tryTapLengths) {
      cout << "Trying tap count [" << tapLength << "]" << endl;
      status = remezGenerateLowPassTaps(
          sampleFrequency,
          cutoffFrequency,
          transitionWidth,
          dbAttenuation,
          taps.data(),
          tapLength);

      if (status == RemezSuccess) {
        cout << "Succeeded with [" << tapLength << "]" << endl;
        taps.resize(tapLength);
        break;
      } else {
        cout << "Failed with [" << tapLength << "]" << endl;
      }
    }

    GS_REQUIRE_OR_RET_FMT(
        status == RemezSuccess,
        Status_RuntimeError,
        "Failed to generate low-pass taps: %s",
        remezStatusToString(status));

    taps.resize(taps.size());

    tapsOut->resize(taps.size());
    for (size_t i = 0; i < tapsOut->size(); i++) {
      tapsOut->at(i) = static_cast<float>(taps[i]);
    }
  }
  IF_CATCH_RETURN_STATUS;

  return Status_Success;
}

static float getQuadDemodGain(float inputSampleRate, float channelWidth) {
  return inputSampleRate / (2.0f * M_PIf * channelWidth);
}

RfToPcmAudioFactory::RfToPcmAudioFactory(IFactories* factories) noexcept
    : mFactories(factories) {}

Result<Node> RfToPcmAudioFactory::create(const char* jsonParameters) noexcept {
  json params;
  UNWRAP_OR_FWD_RESULT(params, parseJson(jsonParameters));

  string modulation = params["modulation"].get<string>();
  const float fskDeviationIfFm = modulation == "fm" ? params["fskDeviation"].get<float>() : 0.0f;

  return ResultCast<Node>(createRfToPcm(
      params["rfSampleRate"],
      params["modulation"],
      params["rfLowPassDecimation"],
      params["audioLowPassDecimation"],
      params["tunedFrequency"],
      params["channelFrequency"],
      params["channelWidth"],
      fskDeviationIfFm,
      params["rfLowPassDbAttenuation"],
      params["audioLowPassDbAttenuation"],
      params["commandQueue"].get<std::string>().c_str()));
}

Result<Filter> RfToPcmAudioFactory::createRfToPcm(
    float rfSampleRate,
    Modulation modulation,
    size_t rfLowPassDecimation,
    size_t audioLowPassDecimation,
    float tunedFrequency,
    float channelFrequency,
    float channelWidth,
    float fskDeviationIfFm,
    float rfLowPassDbAttenuation,
    float audioLowPassDbAttenuation,
    const char* commandQueueId) noexcept {
  const float audioSampleRate =
      rfSampleRate / static_cast<float>(rfLowPassDecimation) / static_cast<float>(audioLowPassDecimation);
  const float quadDemodInputSampleRate = rfSampleRate / static_cast<float>(rfLowPassDecimation);
  const float rfLowPassCutoffFrequency = quadDemodInputSampleRate / 2.0f * 0.95f;
  const float rfLowPassTransitionWidth = quadDemodInputSampleRate / 2.0f * 0.05f;
  const float audioLowPassCutoffFrequency = audioSampleRate / 2.0f * 0.9f;
  const float audioLowPassTransitionWidth = audioSampleRate / 2.0f * 0.1f;

  vector<float> rfLowPassTaps;
  FWD_IN_RESULT_IF_ERR(createLowPassTaps(
      &rfLowPassTaps,
      rfSampleRate,
      rfLowPassCutoffFrequency,
      rfLowPassTransitionWidth,
      rfLowPassDbAttenuation));

  vector<float> audioLowPassTaps;
  FWD_IN_RESULT_IF_ERR(createLowPassTaps(
      &audioLowPassTaps,
      audioSampleRate,
      audioLowPassCutoffFrequency,
      audioLowPassTransitionWidth,
      audioLowPassDbAttenuation));

  gslogd("Command Queue [%s]", commandQueueId);
  gslogd(
      "Channel frequency [%f], tuned frequency [%f] channel width [%f]",
      channelFrequency,
      tunedFrequency,
      channelWidth);
  gslogd("RF sample rate [%f]", rfSampleRate);
  gslogd("Audio sample rate [%ld]", lrint(audioSampleRate));
  gslogd(
      "RF Low-pass cutoff [%f] transition [%f] attenuation [%f] decimation [%zu] tap length [%zu]",
      rfLowPassCutoffFrequency,
      rfLowPassTransitionWidth,
      rfLowPassDbAttenuation,
      rfLowPassDecimation,
      rfLowPassTaps.size());

  gslogd(
      "Audio Low-pass cutoff [%f] transition [%f] attenuation [%f] decimation [%zu] tap length [%zu]",
      audioLowPassCutoffFrequency,
      audioLowPassTransitionWidth,
      audioLowPassDbAttenuation,
      audioLowPassDecimation,
      audioLowPassTaps.size());
  gslogd("Cosine source frequency [%f]", tunedFrequency - channelFrequency);
  gslogd("Quad demod gain [%f]", getQuadDemodGain(quadDemodInputSampleRate, channelWidth));

  json component = {{
      "nodes",
      {
          {
              "cosineSource",
              {
                  {"type", "Cosine"},
                  {"description", "Produce a cosine signal, used to shift the signal's frequency"},
                  {"sampleType", "FloatComplex"},
                  {"sampleRate", rfSampleRate},
                  {"frequency", tunedFrequency - channelFrequency},
                  {"commandQueueId", commandQueueId},
              },
          },
          {
              "multiplyForFrequencyShift",
              {
                  {"type", "Multiply"},
                  {"description", "Multiply the signal by the cosine source to shift the frequency"},
                  {"inputSampleTypes", json::array({"ComplexFloat", "ComplexFloat"})},
              },
          },
          {
              "rfLowPassFilter",
              {
                  {"type", "Fir"},
                  {"description", "Filter out higher frequencies than the shifted frequency"},
                  {"taps", rfLowPassTaps},
                  {"tapType", "Float"},
                  {"signalType", "ComplexFloat"},
                  {"decimation", rfLowPassDecimation},
                  {"commandQueueId", commandQueueId},
              },
          },
          {
              "quadDemod",
              {
                  {"type", "QuadDemod"},
                  {"description", "Demodulate the signal"},
                  {"modulation", modulation},
                  {"sampleRate", quadDemodInputSampleRate},
                  {"fskDeviation", fskDeviationIfFm},
                  {"commandQueueId", commandQueueId},
              },
          },
          {
              "audioLowPassFilter",
              {{"type", "Fir"},
               {"description", "Reduce the audio sample rate for output"},
               {"taps", audioLowPassTaps},
               {"tapType", "Float"},
               {"signalType", "Float"},
               {"decimation", audioLowPassDecimation},
               {"commandQueueId", commandQueueId}},
          },
      },
      {
          "connections",
          json::array({
              {
                  {"source", "cosineSource"},
                  {"sink", "multiplyForFrequencyShift"},
                  {"sinkPort", 1},
              },
              {
                  {"source", "multiplyForFrequencyShift"},
                  {"sink", "rfLowPassFilter"},
              },
              {
                  {"source", "rfLowPassFilter"},
                  {"sink", "quadDemod"},
              },
              {
                  {"source", "quadDemod"},
                  {"sink", "audioLowPassFilter"},
              },
          }),
      },
      {
          "inputPorts",
          json::array({{
              {"exposedPort", 0},
              {"mapped",
               {
                   {"node", "multiplyForFrequencyShift"},
                   {"port", 0},
               }},
          }}),
      },
      {"outputPort", "audioLowPassFilter"},
  }};

  Ref<Node> componentNode;
  UNWRAP_OR_FWD_RESULT(componentNode, createFilter("Component", component.dump().c_str()));

  ConstRef<Filter> componentFilter = componentNode->asFilter();
  if (componentFilter == nullptr) {
    gsloge("RF -> PCM Audio component is not a filter");

    return ERR_RESULT(Status_RuntimeError);
  }

  return makeRefResultNonNull(componentFilter);
}
