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

#include "RfToPcmAudio.h"

#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

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

Result<Filter> RfToPcmAudio::create(
    float rfSampleRate,
    Modulation modulation,
    size_t rfLowPassDecim,
    size_t audioLowPassDecim,
    float centerFrequency,
    float channelFrequency,
    float channelWidth,
    float fskDevationIfFm,
    float rfLowPassDbAttenuation,
    float audioLowPassDbAttenuation,
    int32_t cudaDevice,
    cudaStream_t cudaStream,
    IFactories* factories) noexcept {
  const float audioSampleRate =
      rfSampleRate / static_cast<float>(rfLowPassDecim) / static_cast<float>(audioLowPassDecim);
  const float quadDemodInputSampleRate = rfSampleRate / static_cast<float>(rfLowPassDecim);
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

  gslogd("CUDA device [%d] stream [%p]", cudaDevice, cudaStream);
  gslogd(
      "Channel frequency [%f], center frequency [%f] channel width [%f]",
      channelFrequency,
      centerFrequency,
      channelWidth);
  gslogd("RF sample rate [%f]", rfSampleRate);
  gslogd("Audio sample rate [%ld]", lrint(audioSampleRate));
  gslogd(
      "RF Low-pass cutoff [%f] transition [%f] attenuation [%f] decimation [%zu] tap length [%zu]",
      rfLowPassCutoffFrequency,
      rfLowPassTransitionWidth,
      rfLowPassDbAttenuation,
      rfLowPassDecim,
      rfLowPassTaps.size());

  gslogd(
      "Audio Low-pass cutoff [%f] transition [%f] attenuation [%f] decimation [%zu] tap length [%zu]",
      audioLowPassCutoffFrequency,
      audioLowPassTransitionWidth,
      audioLowPassDbAttenuation,
      audioLowPassDecim,
      audioLowPassTaps.size());
  gslogd("Cosine source frequency [%f]", centerFrequency - channelFrequency);
  gslogd("Quad demod gain [%f]", getQuadDemodGain(quadDemodInputSampleRate, channelWidth));

  Ref<IFilterDriver> driver;
  UNWRAP_OR_FWD_RESULT(driver, factories->getFilterDriverFactory()->createFilterDriver());

  Ref<Source> cosineSource;
  UNWRAP_OR_FWD_RESULT(
      cosineSource,
      factories->getCosineSourceFactory()->createCosineSource(
          SampleType_FloatComplex,
          rfSampleRate,
          centerFrequency - channelFrequency,
          cudaDevice,
          cudaStream));

  Ref<Filter> multiplyRfSourceByCosine;
  UNWRAP_OR_FWD_RESULT(multiplyRfSourceByCosine, factories->getMultiplyFactory()->createFilter(cudaDevice, cudaStream));

  Ref<Filter> rfLowPassFilter;
  UNWRAP_OR_FWD_RESULT(
      rfLowPassFilter,
      factories->getFirFactory()->createFir(
          SampleType_Float,
          SampleType_FloatComplex,
          rfLowPassDecim,
          rfLowPassTaps.data(),
          rfLowPassTaps.size(),
          cudaDevice,
          cudaStream));

  Ref<Filter> quadDemodFilter;
  UNWRAP_OR_FWD_RESULT(
      quadDemodFilter,
      factories->getQuadDemodFactory()->create(modulation, rfSampleRate, fskDevationIfFm, cudaDevice, cudaStream));

  Ref<Filter> audioLowPassFilter;
  UNWRAP_OR_FWD_RESULT(
      audioLowPassFilter,
      factories->getFirFactory()->createFir(
          SampleType_Float,
          SampleType_Float,
          audioLowPassDecim,
          audioLowPassTaps.data(),
          audioLowPassTaps.size(),
          cudaDevice,
          cudaStream));

  Ref<IPortRemappingSink> multiplyWithOnlyPort0Exposed;
  UNWRAP_OR_FWD_RESULT(
      multiplyWithOnlyPort0Exposed,
      factories->getPortRemappingSinkFactory()->create(multiplyRfSourceByCosine.get()));
  multiplyWithOnlyPort0Exposed->addPortMapping(0, 0);

  driver->setDriverInput(multiplyWithOnlyPort0Exposed.get());
  driver->setDriverOutput(audioLowPassFilter.get());

  FWD_IN_RESULT_IF_ERR(driver->connect(cosineSource.get(), 0, multiplyRfSourceByCosine.get(), 1));
  FWD_IN_RESULT_IF_ERR(driver->connect(multiplyRfSourceByCosine.get(), 0, rfLowPassFilter.get(), 0));
  FWD_IN_RESULT_IF_ERR(driver->connect(rfLowPassFilter.get(), 0, quadDemodFilter.get(), 0));
  FWD_IN_RESULT_IF_ERR(driver->connect(quadDemodFilter.get(), 0, audioLowPassFilter.get(), 0));

  FWD_IN_RESULT_IF_ERR(
      driver->setupNode(multiplyWithOnlyPort0Exposed.get(), "Receive RF input, send to Multiply port 0"));
  FWD_IN_RESULT_IF_ERR(driver->setupNode(cosineSource.get(), "Produce a cosine signal"));
  FWD_IN_RESULT_IF_ERR(driver->setupNode(multiplyRfSourceByCosine.get(), "Multiply RF input by cosine"));
  FWD_IN_RESULT_IF_ERR(driver->setupNode(rfLowPassFilter.get(), "Shift frequency with low-pass+decimate RF signal"));
  FWD_IN_RESULT_IF_ERR(driver->setupNode(quadDemodFilter.get(), "Demodulate FM from Quadrature input"));
  FWD_IN_RESULT_IF_ERR(
      driver->setupNode(audioLowPassFilter.get(), "Low-pass and decimate to output audio sample rate"));

  auto rfToPcmAudio = new (nothrow) RfToPcmAudio(
      driver.get(),
      cosineSource.get(),
      multiplyRfSourceByCosine.get(),
      rfLowPassFilter.get(),
      quadDemodFilter.get(),
      audioLowPassFilter.get());

  return makeRefResultNonNull<Filter>(rfToPcmAudio);
}

RfToPcmAudio::RfToPcmAudio(
    IFilterDriver* driver,
    Source* cosineSource,
    Filter* multiply,
    Filter* rfLowPassFilter,
    Filter* quadDemodFilter,
    Filter* audioLowPassFilter) noexcept
    : mDriver(driver),
      mCosineSource(cosineSource),
      mMultiplyRfSourceByCosine(multiply),
      mRfLowPassFilter(rfLowPassFilter),
      mQuadDemod(quadDemodFilter),
      mAudioLowPassFilter(audioLowPassFilter) {}

Result<IBuffer> RfToPcmAudio::requestBuffer(size_t port, size_t byteCount) noexcept {
  return mDriver->requestBuffer(port, byteCount);
}

Status RfToPcmAudio::commitBuffer(size_t port, size_t byteCount) noexcept {
  return mDriver->commitBuffer(port, byteCount);
}

size_t RfToPcmAudio::getOutputDataSize(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
  return mDriver->getOutputDataSize(port);
}

size_t RfToPcmAudio::getOutputSizeAlignment(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
  return mDriver->getOutputSizeAlignment(port);
}

Status RfToPcmAudio::readOutput(IBuffer** portOutputBuffers, size_t numPorts) noexcept {
  return mDriver->readOutput(portOutputBuffers, numPorts);
}

size_t RfToPcmAudio::preferredInputBufferSize(size_t port) noexcept {
  return 1 << 20;
}
