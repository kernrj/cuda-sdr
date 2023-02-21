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
#include <cufft.h>
#include <cufftXt.h>
#include <gpusdrpipeline/CudaErrors.h>
#include <gpusdrpipeline/Factories.h>
#include <gpusdrpipeline/GSLog.h>
#include <gpusdrpipeline/am.h>
#include <gpusdrpipeline/filters/Filter.h>
#include <gpusdrpipeline/fm.h>
#include <remez/remez.h>

#include <atomic>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stack>

using namespace std;

static stack<function<void()>> runAtExit;
static atomic_bool wasInterrupted(false);
static condition_variable wasInterruptedCv;

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

  gslog(GSLOG_INFO, "Caught signal %d, cleaning up.", signum);
  cleanupThings();

  if (signum == SIGINT || signum == SIGTERM) {
    wasInterrupted.store(true);
    wasInterruptedCv.notify_all();
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

static string createIndent(size_t indentSpaces) { return string(indentSpaces, ' '); }

/**
 * https://tomroelandts.com/articles/how-to-create-a-configurable-filter-using-a-kaiser-window
 *
 * @param dbAttenuation
 * @param transitionWidthNormalized
 * @return
 */
static size_t kaiserWindowLength(float dbAttenuation, float transitionWidthNormalized) {
  const size_t windowLength =
      lrintf(ceilf((dbAttenuation - 8.0f) / (2.285f * 2.0f * M_PIf * transitionWidthNormalized) + 1))
      | 1;  // | 1 to make it odd if even.

  return windowLength;
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

  auto driver = unwrap(factories->getFilterDriverFactory()->createFilterDriver());
  THROW_IF_ERR(driver->connect(fileReader, 0, hostToDevice, 0));

  driver->setupNode(fileReader, "Read quadrature samples from file");
  driver->setupNode(hostToDevice, "Copy quadrature samples to GPU");

  driver->setDriverOutput(hostToDevice);

  gslog(GSLOG_INFO, "Created file-based input from [%s]", fileName);

  return ImmutableRef<Source>(driver);
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

  driver->setupNode(hackrfInput, "Get HackRF samples");
  driver->setupNode(hostToDevice, "Copy HackRF samples to GPU memory");
  driver->setupNode(int8ToFloat, "Convert HackRF complex int8 format to complex float");

  runAtExit.emplace([hackrfInput]() { hackrfInput->releaseDevice(); });

  return ImmutableRef<Source>(driver);
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

  driver->setupNode(deviceToHost, "Copy audio samples from GPU to System Memory");
  driver->setupNode(audioOutput, "Encode and mux audio to a file");

  *outputByteCountWritten = deviceToHost;

  return ImmutableRef<Sink>(driver);
}

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

static float getQuadDemodGain(float inputSampleRate, float deviation) {
  return inputSampleRate / (2.0f * M_PIf * deviation);
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
      (channelWidth / 0.95f) * 2.0f,
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
  driver->setupNode(cosineSource, SSTREAM("Produce a cosine signal [" << name << "]").c_str());
  driver->setupNode(multiplyRfSourceByCosine, SSTREAM("Multiply signals [" << name << "]").c_str());
  driver->setupNode(lowPassFilter, SSTREAM("Low-pass filter to smooth out frequency shift [" << name << "]").c_str());

  driver->setDriverInput(multiplyWithOnlyPort0Exposed);
  driver->setDriverOutput(lowPassFilter);

  THROW_IF_ERR(driver->connect(cosineSource, 0, multiplyRfSourceByCosine, 1));
  THROW_IF_ERR(driver->connect(multiplyRfSourceByCosine, 0, lowPassFilter, 0));

  return ImmutableRef<Filter>(driver);
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

  const float audioCutoffFrequency = audioSampleRate / 2.0f * 0.95f;
  const float audioTransitionWidth = audioSampleRate / 2.0f * 0.05f;
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

  driver->setupNode(rfFrequencyShifter, "Shift RF frequency of channel");
  driver->setupNode(quadDemod, "Demodulate FM from Quadrature input");
  driver->setupNode(audioFilter, "Process audio");

  return ImmutableRef<Filter>(driver);
}

int main(int argc, char** argv) {
  atexit(cleanupThings);
  set_terminate(cleanupThings);
  signal(SIGSEGV, &exitSigHandler);
  signal(SIGINT, &exitSigHandler);
  signal(SIGTERM, &exitSigHandler);
  signal(SIGABRT, &exitSigHandler);

  const size_t stopAfterOutputAudioSampleCount = 48000 * 60;
  /*
    auto nbfmTest = NbfmTest();
    nbfmTest.start();
    nbfmTest.doFilter(stopAfterOutputAudioSampleCount);
    nbfmTest.stop();
    */

  /* Lots of static, but can hear the signal
   * auto factories = getFactoriesSingleton();
const auto modulation = Modulation_Fm;
const size_t audioSampleRate = 24000;
const size_t rfLowPassDecim = 4;
const size_t audioLowPassDecim = 10;
const float rfSampleRate = 1e6;
const auto tunedCenterFrequency = 145e6;
const auto channelFrequency = 145.45e6;
const auto rfLowPassDbAttenuation = -60.0f;
const auto audioLowPassDbAttenuation = -60.0f;
const auto channelWidth = 50e3;
const auto fskDevationIfFm = 15e3;
const int32_t cudaDevice = 0;
cudaStream_t cudaStream = createCudaStream();
const char *const inputFileName = "/home/rick/sdr/digital.center145e6.signal145_45e6.iq";
const char* const outputFileName = "/home/rick/sdr/from-digital.ts";
const size_t outputBitRate = 128000;
   */

  auto factories = unwrap(getFactoriesSingleton());
  const auto modulation = Modulation_Fm;
  const size_t audioSampleRate = 8000;
  const size_t rfLowPassDecim = 12;
  const size_t audioLowPassDecim = 10;
  const float rfSampleRate = 1e6;
  const auto tunedCenterFrequency = 145e6;
  const auto channelFrequency = 145.45e6;
  const auto rfLowPassDbAttenuation = -60.0f;
  const auto audioLowPassDbAttenuation = -60.0f;
  const auto channelWidth = kNbfmChannelWidth;
  const auto fskDevationIfFm = kNbfmFrequencyDeviation;
  const int32_t cudaDevice = 0;
  cudaStream_t cudaStream = createCudaStream();
  const char* const inputFileName = "/home/rick/sdr/digital.center145e6.signal145_45e6.iq";
  const char* const outputFileName = "/home/rick/sdr/from-digital.ts";
  const size_t outputBitRate = 128000;

  ImmutableRef<Source> inputPipeline = createQuadFileInputPipeline(inputFileName, factories, cudaDevice, cudaStream);
  Ref<IReadByteCountMonitor> readByteCountMonitor;
  ImmutableRef<Sink> outputPipeline = createAacFileOutputPipeline(
      factories,
      &readByteCountMonitor,
      outputFileName,
      audioSampleRate,
      outputBitRate,
      cudaDevice,
      cudaStream);

  auto rfToAudio = createRfToAudioPipeline(
      rfSampleRate,
      modulation,
      rfLowPassDecim,
      audioLowPassDecim,
      tunedCenterFrequency,
      channelFrequency,
      channelWidth,
      fskDevationIfFm,
      rfLowPassDbAttenuation,
      audioLowPassDbAttenuation,
      cudaDevice,
      cudaStream,
      factories);
  auto audioBandPass = createBandPassFilter(
      SampleType_Float,
      SampleType_Float,
      audioSampleRate,
      975.0f,
      1950.0f,
      100.0f,
      -60.0,
      1,
      factories,
      cudaDevice,
      cudaStream);
  auto audioPitchShift = createFrequencyShifter(
      SampleType_Float,
      audioSampleRate,
      -800.0f,
      3000.0f,
      1,
      "Audio Pitch Shifter",
      factories,
      cudaDevice,
      cudaStream);
  const auto audioLowPassOnShiftedPitch = createLowPassFilter(
      SampleType_Float,
      SampleType_Float,
      audioSampleRate,
      2000.0f,
      100.0f,
      -60,
      1,
      factories,
      cudaDevice,
      cudaStream);

  auto driver = unwrap(factories->getSteppingDriverFactory()->createSteppingDriver());

  THROW_IF_ERR(driver->connect(inputPipeline, 0, rfToAudio, 0));
  THROW_IF_ERR(driver->connect(rfToAudio, 0, audioBandPass, 0));
  THROW_IF_ERR(driver->connect(audioBandPass, 0, audioPitchShift, 0));
  THROW_IF_ERR(driver->connect(audioPitchShift, 0, audioLowPassOnShiftedPitch, 0));
  THROW_IF_ERR(driver->connect(audioLowPassOnShiftedPitch, 0, outputPipeline, 0));

  driver->setupNode(inputPipeline, "RF Input Pipeline");
  driver->setupNode(rfToAudio, "Convert RF signal to audio");
  driver->setupNode(audioBandPass, "Filter audio frequencies");
  driver->setupNode(outputPipeline, "Audio Output Pipeline");
  driver->setupNode(audioPitchShift, "Audio Pitch Shift");
  driver->setupNode(audioLowPassOnShiftedPitch, "Remove high frequencies from pitch shift");

  const size_t maxSampleCount = audioSampleRate * 60;

  auto driverToDot = unwrap(factories->getDriverToDotFactory()->create());
  size_t dotGraphSize =
      unwrap(driverToDot->convertToDot(driver.get(), "Converts a digital signal to audio file", nullptr, 0));

  vector<char> dotGraph(dotGraphSize + 1);
  unwrap(driverToDot->convertToDot(
      driver.get(),
      "Converts a digital signal to audio file",
      dotGraph.data(),
      dotGraph.capacity()));

  gslog(GSLOG_INFO, "%s", dotGraph.data());

  size_t itCount = 0;
  while (readByteCountMonitor->getByteCountRead(0) / sizeof(float) < maxSampleCount) {
    driver->doFilter();

    /*
    itCount++;

    if (itCount == 100) {
      break;
    }*/
  }

  return 0;
}
