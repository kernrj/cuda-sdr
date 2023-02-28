/*
 * Copyright 2022 Rick Kern <kernrj@gmail.com>
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

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <gpusdrpipeline/CudaErrors.h>
#include <gpusdrpipeline/Factories.h>
#include <gpusdrpipeline/am.h>
#include <remez/remez.h>

#include <atomic>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <stack>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

using namespace std;

static stack<function<void()>> runAtExit;
static atomic_bool wasInterrupted(false);

static constexpr bool useFileInput = false;
static bool aborted = false;

static void cleanupThings() {
  clog << "Cleaning up - shutting down" << endl;

  while (!runAtExit.empty()) {
    const auto toRun = std::move(runAtExit.top());
    runAtExit.pop();

    toRun();
  }
}

static void exitSigHandler(int signum) {
  if (aborted) {
    return;
  }

  gslogi("Caught signal %d, cleaning up.", signum);
  cleanupThings();

  if (signum == SIGINT || signum == SIGTERM) {
    wasInterrupted.store(true);
  } else {
    aborted = true;
    abort();
  }
}

static cudaStream_t createCudaStream() {
  cudaStream_t cudaStream = nullptr;
  SAFE_CUDA_OR_THROW(cudaStreamCreate(&cudaStream));

  return cudaStream;
}

static float getQuadDemodGain(float inputSampleRate, float deviation) {
  return inputSampleRate / (2.0f * M_PIf * deviation);
}

static ImmutableRef<Source> createQuadFileInputPipeline(
    const char* fileName,
    IFactories* factories,
    int32_t cudaDevice,
    cudaStream_t cudaStream) {
  const auto fileReader = unwrap(factories->getFileReaderFactory()->createFileReader(fileName));
  const auto cudaAlloc =
      unwrap(factories->getCudaAllocatorFactory()->createCudaAllocator(cudaDevice, cudaStream, 32, false));
  const auto hostToDevice =
      unwrap(factories->getCudaMemcpyFilterFactory()->createCudaMemcpy(cudaMemcpyHostToDevice, cudaDevice, cudaStream));

  ConstRef<IFilterDriver> driver = unwrap(factories->getFilterDriverFactory()->createFilterDriver());
  THROW_IF_ERR(driver->connect(fileReader, 0, hostToDevice, 0));

  THROW_IF_ERR(driver->setupNode(fileReader, "Read quadrature samples from file"));
  THROW_IF_ERR(driver->setupNode(hostToDevice, "Copy quadrature samples to GPU"));

  driver->setDriverOutput(hostToDevice);

  gslogi("Created file-based input from [%s]", fileName);
  ImmutableRef<Source> driverAsSource = driver.get();

  return driverAsSource;
}

static ImmutableRef<Sink> createAacFileOutputPipeline(
    IFactories* factories,
    Ref<IReadByteCountMonitor>* outputByteCountWritten,
    const char* outputFileName,
    size_t audioSampleRate,
    size_t outputBitRate,
    int32_t cudaDevice,
    cudaStream_t cudaStream) {
  auto deviceToHost = unwrap(factories->getReadByteCountMonitorFactory()->create(unwrap(
      factories->getCudaMemcpyFilterFactory()->createCudaMemcpy(cudaMemcpyDeviceToHost, cudaDevice, cudaStream))));
  auto audioOutput =
      unwrap(factories->getAacFileWriterFactory()
                 ->createAacFileWriter(outputFileName, audioSampleRate, outputBitRate, cudaDevice, cudaStream));

  auto driver = unwrap(factories->getFilterDriverFactory()->createFilterDriver());

  driver->setDriverInput(deviceToHost);
  THROW_IF_ERR(driver->connect(deviceToHost, 0, audioOutput, 0));

  THROW_IF_ERR(driver->setupNode(deviceToHost, "Copy audio samples from GPU to System Memory"));
  THROW_IF_ERR(driver->setupNode(audioOutput, "Encode and mux audio to a file"));

  *outputByteCountWritten = deviceToHost;

  return ImmutableRef<Sink>(driver.get());
}

static size_t bellangerLowPassTapCount(
    float sampleRateHz,
    float transitionWidthHz,
    float passBandRipple,
    float dbAttenuation) {
  GS_REQUIRE_LTE(dbAttenuation, 0, "Attenuation not be positive, but is [" << dbAttenuation << "]");

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

static vector<float> createBandPassTaps(
    float sampleFrequency,
    float cutoffFrequencyLow,
    float cutoffFrequencyHigh,
    float transitionWidth,
    float dbAttenuation) {
  const size_t bellangerTapCount = bellangerLowPassTapCount(sampleFrequency, transitionWidth, 0.01f, dbAttenuation) * 2;
  const size_t fredHarrisTapCount = fredHarrisLowPassTapCount(dbAttenuation, transitionWidth, sampleFrequency) * 2;
  cout << "Bellanger [" << bellangerTapCount << "] fred harris [" << fredHarrisTapCount << "]" << endl;

  vector<size_t> tryTapLengths = {fredHarrisTapCount, bellangerTapCount, fredHarrisTapCount / 2, bellangerTapCount / 2};

  auto maxSizeIt = max_element(begin(tryTapLengths), end(tryTapLengths));
  GS_REQUIRE_OR_THROW(maxSizeIt != tryTapLengths.end(), "Empty length vector computing low-pass taps");
  const size_t maxSize = *maxSizeIt;

  vector<double> taps(maxSize);
  cout << "Creating low-pass taps with sample frequency [" << sampleFrequency << "] cutoff low [" << cutoffFrequencyLow
       << "] cutoff high [" << cutoffFrequencyHigh << "] transition width [" << transitionWidth << "] attenuation ["
       << dbAttenuation << "] taps size [" << taps.size() << "]" << endl;

  RemezStatus status = RemezSuccess;

  for (size_t tapLength : tryTapLengths) {
    cout << "Trying tap count [" << tapLength << "]" << endl;
    status = remezGenerateSingleBandPassTaps(
        sampleFrequency,
        cutoffFrequencyLow,
        cutoffFrequencyHigh,
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

  if (status != RemezSuccess) {
    GS_FAIL("Failed to generate low-pass taps: " << remezStatusToString(status));
  }

  taps.resize(taps.size());

  vector<float> floatTaps(taps.size());
  for (size_t i = 0; i < floatTaps.size(); i++) {
    floatTaps[i] = static_cast<float>(taps[i]);
  }

  return floatTaps;
}

static vector<float> createLowPassTaps(
    float sampleFrequency,
    float cutoffFrequency,
    float transitionWidth,
    float dbAttenuation) {
  const size_t bellangerTapCount = bellangerLowPassTapCount(sampleFrequency, transitionWidth, 0.01f, dbAttenuation);
  const size_t fredHarrisTapCount = fredHarrisLowPassTapCount(dbAttenuation, transitionWidth, sampleFrequency);
  cout << "Bellanger [" << bellangerTapCount << "] fred harris [" << fredHarrisTapCount << "]" << endl;

  vector<size_t> tryTapLengths = {fredHarrisTapCount, bellangerTapCount, fredHarrisTapCount / 2, bellangerTapCount / 2};

  auto maxSizeIt = max_element(begin(tryTapLengths), end(tryTapLengths));
  GS_REQUIRE_OR_THROW(maxSizeIt != tryTapLengths.end(), "Empty length vector computing low-pass taps");
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

  if (status != RemezSuccess) {
    GS_FAIL("Failed to generate low-pass taps: " << remezStatusToString(status));
  }

  taps.resize(taps.size());

  vector<float> floatTaps(taps.size());
  for (size_t i = 0; i < floatTaps.size(); i++) {
    floatTaps[i] = static_cast<float>(taps[i]);
  }

  return floatTaps;
}

static ImmutableRef<Filter> createLowPassFilter(
    SampleType tapType,
    SampleType elementType,
    float inputSampleRate,
    float cutoffFrequency,
    float transitionWidth,
    float dbAttenuation,
    size_t decimation,
    IFactories* factories,
    int32_t cudaDevice,
    cudaStream_t cudaStream) {
  auto taps = createLowPassTaps(inputSampleRate, cutoffFrequency, transitionWidth, dbAttenuation);
  return unwrap(factories->getFirFactory()
                    ->createFir(tapType, elementType, decimation, taps.data(), taps.size(), cudaDevice, cudaStream));
}

static ImmutableRef<Filter> createBandPassFilter(
    SampleType tapType,
    SampleType elementType,
    float inputSampleRate,
    float cutoffFrequencyLow,
    float cutoffFrequencyHigh,
    float transitionWidth,
    float dbAttenuation,
    size_t decimation,
    IFactories* factories,
    int32_t cudaDevice,
    cudaStream_t cudaStream) {
  auto taps =
      createBandPassTaps(inputSampleRate, cutoffFrequencyLow, cutoffFrequencyHigh, transitionWidth, dbAttenuation);
  return unwrap(factories->getFirFactory()
                    ->createFir(tapType, elementType, decimation, taps.data(), taps.size(), cudaDevice, cudaStream));
}

static ImmutableRef<Filter> createFrequencyShifter(
    SampleType sampleType,
    float inputSampleRate,
    float shiftFrequencyBy,
    float channelWidth,
    size_t lowPassDecimation,
    const char* name,
    IFactories* factories,
    int32_t cudaDevice,
    cudaStream_t cudaStream) {
  float lowPassCutoffFrequencyMax = inputSampleRate / 2.0f * 0.95f;
  GS_REQUIRE_OR_THROW_FMT(
      channelWidth <= lowPassCutoffFrequencyMax,
      "Sample rate [%f] is too low for a channel width of [%f]. "
      "The sample rate must be at least [%f] or the channel width must be at most [%f]",
      inputSampleRate,
      channelWidth,
      channelWidth / 0.95f * 2.0f,
      lowPassCutoffFrequencyMax);
  float lowPassCutoffFrequency = channelWidth / 2;
  float lowPassTransitionWidth = min(channelWidth / 4, inputSampleRate / 2.0f - lowPassCutoffFrequency);
  float lowPassDbAttenuation = -60.0f;

  auto cosineSource =
      unwrap(factories->getCosineSourceFactory()
                 ->createCosineSource(sampleType, inputSampleRate, shiftFrequencyBy, cudaDevice, cudaStream));
  auto multiplyRfSourceByCosine = unwrap(factories->getMultiplyFactory()->createFilter(cudaDevice, cudaStream));
  auto lowPassFilter = createLowPassFilter(
      SampleType_Float,
      SampleType_FloatComplex,
      inputSampleRate,
      lowPassCutoffFrequency,
      lowPassTransitionWidth,
      lowPassDbAttenuation,
      lowPassDecimation,
      factories,
      cudaDevice,
      cudaStream);

  const auto multiplyWithOnlyPort0Exposed =
      unwrap(factories->getPortRemappingSinkFactory()->create(multiplyRfSourceByCosine));
  multiplyWithOnlyPort0Exposed->addPortMapping(0, 0);

  auto driver = unwrap(factories->getFilterDriverFactory()->createFilterDriver());
  THROW_IF_ERR(driver->setupNode(cosineSource, SSTREAM("Produce a cosine signal [" << name << "]").c_str()));
  THROW_IF_ERR(driver->setupNode(multiplyRfSourceByCosine, SSTREAM("Multiply signals [" << name << "]").c_str()));
  THROW_IF_ERR(driver->setupNode(
      lowPassFilter,
      SSTREAM("Low-pass filter to smooth out frequency shift [" << name << "]").c_str()));

  driver->setDriverInput(multiplyWithOnlyPort0Exposed);
  driver->setDriverOutput(lowPassFilter);

  THROW_IF_ERR(driver->connect(cosineSource, 0, multiplyRfSourceByCosine, 1));
  THROW_IF_ERR(driver->connect(multiplyRfSourceByCosine, 0, lowPassFilter, 0));

  return ImmutableRef<Filter>(driver.get());
}

static ImmutableRef<Source> createHackrfInputPipeline(
    IFactories* factories,
    float tunedCenterFrequency,
    float rfSampleRate,
    int32_t cudaDevice,
    cudaStream_t cudaStream) {
  const size_t maxHackrfBuffersBeforeDropping = 3;
  auto hackrfInput =
      unwrap(factories->getHackrfSourceFactory()
                 ->createHackrfSource(0, tunedCenterFrequency, rfSampleRate, maxHackrfBuffersBeforeDropping));
  auto hostToDevice =
      unwrap(factories->getCudaMemcpyFilterFactory()->createCudaMemcpy(cudaMemcpyHostToDevice, cudaDevice, cudaStream));
  auto int8ToFloat = unwrap(factories->getInt8ToFloatFactory()->createFilter(cudaDevice, cudaStream));

  auto driver = unwrap(factories->getFilterDriverFactory()->createFilterDriver());
  THROW_IF_ERR(driver->connect(hackrfInput, 0, hostToDevice, 0));
  THROW_IF_ERR(driver->connect(hostToDevice, 0, int8ToFloat, 0));
  driver->setDriverOutput(int8ToFloat);

  THROW_IF_ERR(driver->setupNode(hackrfInput, "Get HackRF samples"));
  THROW_IF_ERR(driver->setupNode(hostToDevice, "Copy HackRF samples to GPU memory"));
  THROW_IF_ERR(driver->setupNode(int8ToFloat, "Convert HackRF complex int8 format to complex float"));

  runAtExit.emplace([hackrfInput]() { THROW_IF_ERR(hackrfInput->releaseDevice()); });

  return ImmutableRef<Source>(driver.get());
}

static ImmutableRef<Filter> createRfToAudioPipeline(
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
    IFactories* factories) {
  auto driver = unwrap(factories->getFilterDriverFactory()->createFilterDriver());
  size_t audioSampleRate = rfSampleRate / rfLowPassDecim / audioLowPassDecim;
  float quadDemodInputSampleRate = rfSampleRate / rfLowPassDecim;

  auto rfFrequencyShifter = createFrequencyShifter(
      SampleType_FloatComplex,
      rfSampleRate,
      centerFrequency - channelFrequency,
      channelWidth,
      rfLowPassDecim,
      "RF Input",
      factories,
      cudaDevice,
      cudaStream);

  auto quadDemod = unwrap(
      factories->getQuadDemodFactory()->create(modulation, rfSampleRate, fskDevationIfFm, cudaDevice, cudaStream));

  const float audioCutoffFrequency = audioSampleRate / 2.0f * 0.9f;
  const float audioTransitionWidth = audioSampleRate / 2.0f * 0.1f;
  const auto audioFilter = createLowPassFilter(
      SampleType_Float,
      SampleType_Float,
      quadDemodInputSampleRate,
      audioCutoffFrequency,
      audioTransitionWidth,
      audioLowPassDbAttenuation,
      audioLowPassDecim,
      factories,
      cudaDevice,
      cudaStream);

  driver->setDriverInput(rfFrequencyShifter);
  driver->setDriverOutput(audioFilter);

  THROW_IF_ERR(driver->connect(rfFrequencyShifter, 0, quadDemod, 0));
  THROW_IF_ERR(driver->connect(quadDemod, 0, audioFilter, 0));

  THROW_IF_ERR(driver->setupNode(rfFrequencyShifter, "Shift RF frequency of channel"));
  THROW_IF_ERR(driver->setupNode(quadDemod, "Demodulate FM from Quadrature input"));
  THROW_IF_ERR(driver->setupNode(audioFilter, "Process audio"));

  return ImmutableRef<Filter>(driver);
}

int main(int argc, char** argv) {
  const int32_t cudaDevice = 0;
  const float maxRfSampleRate = 1e6;
  const float audioSampleRate = 8e3;
  const auto rfDecimation = 5;
  const auto audioDecimation = 25;
  const float rfLowPassDbAttenuation = -60.0f;
  const float audioLowPassDbAttenuation = -60.0f;
  const float rfSampleRate = rfDecimation * audioDecimation * audioSampleRate;
  const float centerFrequency = 2000e3;
  const float channelFrequency = 1340e3;
  const float channelWidth = kAmChannelBandwidth;
  const float channelPassBandWidth = 9.69e3;
  const float channelEndBandWidth = 10.15e3;
  const float cutoffFrequency = channelPassBandWidth / 2.0f;
  const float transitionWidth = (channelEndBandWidth - channelPassBandWidth) / 2.0f;
  const char* const outputFileName = "/home/rick/sdr/am.ts";
  const size_t outputBitRate = 128000;

  cudaStream_t cudaStream = nullptr;
  SAFE_CUDA_OR_THROW(cudaStreamCreate(&cudaStream));
  ConstRef<IFactories> factories = unwrap(getFactoriesSingleton());
  const auto inputPipeline =
      createHackrfInputPipeline(factories, centerFrequency, rfSampleRate, cudaDevice, cudaStream);

  const auto rfToAudio = createRfToAudioPipeline(
      rfSampleRate,
      Modulation_Am,
      rfDecimation,
      audioDecimation,
      centerFrequency,
      channelFrequency,
      channelWidth,
      0.0f,
      rfLowPassDbAttenuation,
      audioLowPassDbAttenuation,
      cudaDevice,
      cudaStream,
      factories);

  Ref<IReadByteCountMonitor> readByteCountMonitor;
  const auto outputPipeline = createAacFileOutputPipeline(
      factories,
      &readByteCountMonitor,
      outputFileName,
      audioSampleRate,
      outputBitRate,
      cudaDevice,
      cudaStream);

  auto driver = unwrap(factories->getSteppingDriverFactory()->createSteppingDriver());

  THROW_IF_ERR(driver->connect(inputPipeline, 0, rfToAudio, 0));
  THROW_IF_ERR(driver->connect(rfToAudio, 0, outputPipeline, 0));

  THROW_IF_ERR(driver->setupNode(inputPipeline, "RF Input Pipeline"));
  THROW_IF_ERR(driver->setupNode(rfToAudio, "Convert RF signal to audio"));
  THROW_IF_ERR(driver->setupNode(outputPipeline, "Audio Output Pipeline"));

  const size_t maxSampleCount = lrint(audioSampleRate) * 10;
  size_t itCount = 0;
  while (readByteCountMonitor->getByteCountRead(0) / sizeof(float) < maxSampleCount) {
    THROW_IF_ERR(driver->doFilter());
  }

  return 0;
}
