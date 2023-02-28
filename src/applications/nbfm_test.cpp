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
#include <gpusdrpipeline/filters/Filter.h>
#include <gpusdrpipeline/fm.h>
#include <gpusdrpipeline/util/CudaDevicePushPop.h>
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

  gslogi("Caught signal %d, cleaning up.", signum);
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

static vector<float> createLowPassTaps(
    float sampleFrequency,
    float cutoffFrequency,
    float transitionWidth,
    float dbAttenuation) {
  const size_t bellangerTapCount = bellangerLowPassTapCount(sampleFrequency, transitionWidth, 0.01f, dbAttenuation);
  const size_t fredHarrisTapCount = fredHarrisLowPassTapCount(dbAttenuation, transitionWidth, sampleFrequency);

  cout << "Bellanger [" << bellangerTapCount << "] fred harris [" << fredHarrisTapCount << "]" << endl;
  const size_t maxRetryCount = 100;
  size_t useTapCount = fredHarrisTapCount;
  vector<double> taps(useTapCount + maxRetryCount);
  cout << "Creating low-pass taps with sample frequency [" << sampleFrequency << "] cutoff [" << cutoffFrequency
       << "] transition width [" << transitionWidth << "] attenuation [" << dbAttenuation << "] taps size ["
       << taps.size() << "]" << endl;

  RemezStatus status = RemezSuccess;

  for (size_t i = 0; i < maxRetryCount; i++) {
    cout << "Trying tap count [" << useTapCount + i << "]" << endl;
    status = remezGenerateLowPassTaps(
        sampleFrequency,
        cutoffFrequency,
        transitionWidth,
        dbAttenuation,
        taps.data(),
        useTapCount + i);

    if (status == RemezSuccess) {
      cout << "Succeeded with [" << useTapCount + i << "]" << endl;
      taps.resize(bellangerTapCount + i);
      break;
    } else {
      cout << "Failed with [" << useTapCount + i << "]" << endl;
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

static float getQuadDemodGain(float inputSampleRate, float channelWidth) {
  return inputSampleRate / (2.0f * M_PIf * channelWidth);
}

class NbfmTest {
 public:
  NbfmTest()
      : factories(unwrap(getFactoriesSingleton())),
        cudaDevice(0),
        cudaStream(createCudaStream()),
        mRfLowPassTaps(createLowPassTaps(
            rfSampleRate,
            rfLowPassCutoffFrequency,
            rfLowPassTransitionWidth,
            rfLowPassDbAttenuation)),
        mAudioLowPassTaps(createLowPassTaps(
            quadDemodInputSampleRate,
            audioLowPassCutoffFrequency,
            audioLowPassTransitionWidth,
            audioLowPassDbAttenuation)),
        hackrfSource(
            unwrap(factories->getHackrfSourceFactory()->createHackrfSource(0, centerFrequency, rfSampleRate, 3))),
        hostToDevice(unwrap(
            factories->getCudaMemcpyFilterFactory()->createCudaMemcpy(cudaMemcpyHostToDevice, cudaDevice, cudaStream))),
        convertHackrfInputToFloat(unwrap(factories->getInt8ToFloatFactory()->createFilter(cudaDevice, cudaStream))),
        cosineSource(unwrap(factories->getCosineSourceFactory()->createCosineSource(
            SampleType_FloatComplex,
            rfSampleRate,
            centerFrequency - channelFrequency,
            cudaDevice,
            cudaStream))),
        multiplyRfSourceByCosine(unwrap(factories->getMultiplyFactory()->createFilter(cudaDevice, cudaStream))),
        rfLowPassFilter(unwrap(factories->getFirFactory()->createFir(
            /* tapType= */ SampleType_Float,
            /* elementType= */ SampleType_FloatComplex,
            rfLowPassDecimation,
            mRfLowPassTaps.data(),
            mRfLowPassTaps.size(),
            cudaDevice,
            cudaStream))),
        quadDemod(unwrap(factories->getQuadDemodFactory()
                             ->create(Modulation_Fm, rfSampleRate, channelWidth, cudaDevice, cudaStream))),
        audioLowPassFilter(unwrap(factories->getFirFactory()->createFir(
            /* tapType= */ SampleType_Float,
            /* elementType= */ SampleType_FloatComplex,
            audioLowPassDecimation,
            mRfLowPassTaps.data(),
            mRfLowPassTaps.size(),
            cudaDevice,
            cudaStream))),
        deviceToHost(unwrap(
            factories->getCudaMemcpyFilterFactory()->createCudaMemcpy(cudaMemcpyDeviceToHost, cudaDevice, cudaStream))),
        floatQuadReader(unwrap(factories->getFileReaderFactory()->createFileReader(inputFileName))),
        audioFileWriter(unwrap(
            factories->getAacFileWriterFactory()
                ->createAacFileWriter(outputAudioFile, audioSampleRate, outputAudioBitRate, cudaDevice, cudaStream))) {
    gslogi("Input file [%s]", inputFileName);
    gslogi("Output file [%s] sample rate [%zu] bit rate [%d]", outputAudioFile, audioSampleRate, outputAudioBitRate);
    gslogi("CUDA device [%d] stream [%p]", cudaDevice, cudaStream);
    gslogi(
        "Channel frequency [%f], center frequency [%f] channel width [%f]",
        channelFrequency,
        centerFrequency,
        channelWidth);
    gslogi("RF sample rate [%f]", rfSampleRate);
    gslogi(
        "RF Low-pass cutoff [%f] transition [%f] attenuation [%f] decimation [%zu] tap length [%zu]",
        rfLowPassCutoffFrequency,
        rfLowPassTransitionWidth,
        rfLowPassDbAttenuation,
        rfLowPassDecimation,
        mRfLowPassTaps.size());

    gslogi(
        "Audio Low-pass cutoff [%f] transition [%f] attenuation [%f] decimation [%zu] tap length [%zu]",
        audioLowPassCutoffFrequency,
        audioLowPassTransitionWidth,
        audioLowPassDbAttenuation,
        audioLowPassDecimation,
        mAudioLowPassTaps.size());
    gslogi("Cosine source frequency [%f]", centerFrequency - channelFrequency);
    gslogi("Quad demod gain [%f]", getQuadDemodGain(quadDemodInputSampleRate, channelWidth));
  }

  void start() {
    if (!useFileInput) {
      THROW_IF_ERR(hackrfSource->start());
    }
  }

  void stop() {
    if (!useFileInput) {
      THROW_IF_ERR(hackrfSource->stop());
    }
  }

  Status doFilter(size_t stopAfterOutputSampleCount) {
    size_t wroteSampleCount = 0;

    CUDA_DEV_PUSH_POP_OR_RET_STATUS(cudaDevice);
    vector<IBuffer*> outputBuffers(1);

    while (wroteSampleCount < stopAfterOutputSampleCount) {
      Source* gpuFloatSource = nullptr;

      if (useFileInput) {
        ConstRef<IBuffer> samples =
            unwrap(hostToDevice->requestBuffer(0, floatQuadReader->getAlignedOutputDataSize(0)));
        outputBuffers[0] = samples;
        THROW_IF_ERR(floatQuadReader->readOutput(outputBuffers.data(), 1));
        THROW_IF_ERR(hostToDevice->commitBuffer(0, samples->range()->used()));

        if (samples->range()->used() == 0) {
          return Status_Success;
        }

        gpuFloatSource = hostToDevice.get();
      } else {
        size_t hackrfBufferLength = hackrfSource->getAlignedOutputDataSize(0);

        ConstRef<IBuffer> hackrfHostSamples = unwrap(hostToDevice->requestBuffer(0, hackrfBufferLength));

        outputBuffers[0] = hackrfHostSamples;
        THROW_IF_ERR(hackrfSource->readOutput(outputBuffers.data(), 1));
        THROW_IF_ERR(hostToDevice->commitBuffer(0, hackrfHostSamples->range()->used()));

        ConstRef<IBuffer> s8ToFloatInputCudaBuffer =
            unwrap(convertHackrfInputToFloat->requestBuffer(0, hostToDevice->getAlignedOutputDataSize(0)));
        outputBuffers[0] = s8ToFloatInputCudaBuffer;
        THROW_IF_ERR(hostToDevice->readOutput(outputBuffers.data(), 1));

        THROW_IF_ERR(convertHackrfInputToFloat->commitBuffer(0, s8ToFloatInputCudaBuffer->range()->used()));
        gpuFloatSource = convertHackrfInputToFloat.get();
      }

      const size_t multiplyInputBufferSize = gpuFloatSource->getAlignedOutputDataSize(0);
      ConstRef<IBuffer> rfMultiplyInput =
          unwrap(multiplyRfSourceByCosine->requestBuffer(0, gpuFloatSource->getAlignedOutputDataSize(0)));
      ConstRef<IBuffer> cosineMultiplyInput =
          unwrap(multiplyRfSourceByCosine->requestBuffer(1, gpuFloatSource->getAlignedOutputDataSize(0)));

      outputBuffers[0] = rfMultiplyInput;
      THROW_IF_ERR(gpuFloatSource->readOutput(outputBuffers.data(), 1));

      if (multiplyInputBufferSize != rfMultiplyInput->range()->used()) {
        GS_FAIL(
            "Expected source complex-float buffer size of [" << multiplyInputBufferSize << "] but got ["
                                                             << rfMultiplyInput->range()->used() << "]");
      }

      THROW_IF_ERR(multiplyRfSourceByCosine->commitBuffer(0, rfMultiplyInput->range()->used()));

      outputBuffers[0] = cosineMultiplyInput;
      THROW_IF_ERR(cosineSource->readOutput(outputBuffers.data(), 1));

      THROW_IF_ERR(multiplyRfSourceByCosine->commitBuffer(1, cosineMultiplyInput->range()->used()));

      ConstRef<IBuffer> rfLowPassBuffer =
          unwrap(rfLowPassFilter->requestBuffer(0, multiplyRfSourceByCosine->getAlignedOutputDataSize(0)));
      outputBuffers[0] = rfLowPassBuffer;
      THROW_IF_ERR(multiplyRfSourceByCosine->readOutput(outputBuffers.data(), 1));
      THROW_IF_ERR(rfLowPassFilter->commitBuffer(0, rfLowPassBuffer->range()->used()));

      ConstRef<IBuffer> quadDemodInputBuffer =
          unwrap(quadDemod->requestBuffer(0, rfLowPassFilter->getAlignedOutputDataSize(0)));
      outputBuffers[0] = quadDemodInputBuffer;
      THROW_IF_ERR(rfLowPassFilter->readOutput(outputBuffers.data(), 1));
      THROW_IF_ERR(quadDemod->commitBuffer(0, quadDemodInputBuffer->range()->used()));

      ConstRef<IBuffer> audioLowPassBuffer =
          unwrap(audioLowPassFilter->requestBuffer(0, quadDemod->getAlignedOutputDataSize(0)));
      outputBuffers[0] = audioLowPassBuffer;
      THROW_IF_ERR(quadDemod->readOutput(outputBuffers.data(), 1));
      THROW_IF_ERR(audioLowPassFilter->commitBuffer(0, audioLowPassBuffer->range()->used()));

      ConstRef<IBuffer> deviceToHostBuffer =
          unwrap(deviceToHost->requestBuffer(0, audioLowPassFilter->getAlignedOutputDataSize(0)));
      outputBuffers[0] = deviceToHostBuffer;
      THROW_IF_ERR(audioLowPassFilter->readOutput(outputBuffers.data(), 1));
      THROW_IF_ERR(deviceToHost->commitBuffer(0, deviceToHostBuffer->range()->used()));

      ConstRef<IBuffer> audioEncMuxInputBuffer =
          unwrap(audioFileWriter->requestBuffer(0, deviceToHost->getAlignedOutputDataSize(0)));

      outputBuffers[0] = audioEncMuxInputBuffer;
      THROW_IF_ERR(deviceToHost->readOutput(outputBuffers.data(), 1));

      cudaStreamSynchronize(cudaStream);

      wroteSampleCount += audioEncMuxInputBuffer->range()->used() / sizeof(float);
      THROW_IF_ERR(audioFileWriter->commitBuffer(0, audioEncMuxInputBuffer->range()->used()));
    }

    return Status_Success;
  }

 private:
  const char* const fileName98_5MHz_Fm_Wb = "/home/rick/sdr/98.5MHz.float.iq";
  const char* const fileName145_45MHz_Fm_Nb = "/home/rick/sdr/raw.bin";
  const char* const inputFileName = fileName145_45MHz_Fm_Nb;

  const char* const outputAudioFile = "/home/rick/sdr/allgpu.ts";

  ConstRef<IFactories> factories;
  const int32_t cudaDevice;
  cudaStream_t cudaStream;
  const vector<float> mRfLowPassTaps;
  const vector<float> mAudioLowPassTaps;
  ConstRef<IHackrfSource> hackrfSource;
  ConstRef<Filter> hostToDevice;
  ConstRef<Filter> convertHackrfInputToFloat;
  ConstRef<Source> cosineSource;
  ConstRef<Filter> multiplyRfSourceByCosine;
  ConstRef<Filter> rfLowPassFilter;
  ConstRef<Filter> quadDemod;
  ConstRef<Filter> audioLowPassFilter;
  ConstRef<Filter> deviceToHost;
  ConstRef<Source> floatQuadReader;
  ConstRef<Sink> audioFileWriter;

  /* 145.45MHz, source file isn't working (not sure of RF sample rate)
  static constexpr float maxRfSampleRate = 20e6;
  static constexpr size_t audioSampleRate = 48e3;
  static constexpr int32_t outputAudioBitRate = 128000;
  static constexpr float centerFrequency = 144e6;
  static constexpr float channelFrequency = 145.45e6;
  static constexpr float rfLowPassDbAttenuation = -60.0f;
  static constexpr float audioLowPassDbAttenuation = -60.0f;
  static constexpr float channelWidth = kNbfmChannelWidth;
  static constexpr float rfSampleRate =
      uint32_t(maxRfSampleRate / audioSampleRate) * audioSampleRate;  // 416 * audio sample rate
  static constexpr size_t audioLowPassDecimation = 26;
  static constexpr size_t rfLowPassDecimation = 16;  // 416 = 16 * 26
  static constexpr float rfLowPassCutoffFrequency = channelWidth;
  static constexpr float rfLowPassTransitionWidth = channelWidth / 2.0f;
  static constexpr size_t quadDemodInputSampleRate = audioSampleRate * audioLowPassDecimation;
*/

  /*
  //98.5MHz channel (WBFM) file capture. RF sample rate 4.8MHz, center frequency 97.5MHz.
  static constexpr size_t audioSampleRate = 48e3;
  static constexpr int32_t outputAudioBitRate = 128000;
  static constexpr float centerFrequency = 97.5e6;
  static constexpr float channelFrequency = 98.5e6;
  static constexpr float rfLowPassDbAttenuation = -60.0f;
  static constexpr float audioLowPassDbAttenuation = -60.0f;
  static constexpr float channelWidth = kWbfmChannelWidth;
  static constexpr size_t audioLowPassDecimation = 10;
  static constexpr size_t rfLowPassDecimation = 10;
  static constexpr size_t quadDemodInputSampleRate = audioSampleRate * audioLowPassDecimation;
  static constexpr float rfSampleRate = static_cast<float>(rfLowPassDecimation) * quadDemodInputSampleRate;
  static constexpr float rfLowPassCutoffFrequency = channelWidth;
  static constexpr float rfLowPassTransitionWidth = channelWidth / 2.0f;
  */

  // 98.5MHz channel (WBFM) live.
  static constexpr float maxRfSampleRate = 19968000.0f;  // Largest multiple of 48KHz less than 20MHz, 416 * 48000
  static constexpr size_t audioSampleRate = 48e3;
  static constexpr int32_t outputAudioBitRate = 128000;
  static constexpr float centerFrequency = 97.5e6;
  static constexpr float channelFrequency = 98.5e6;
  static constexpr float rfLowPassDbAttenuation = -60.0f;
  static constexpr float audioLowPassDbAttenuation = -60.0f;
  static constexpr float channelWidth = kWbfmChannelWidth;
  static constexpr size_t audioLowPassDecimation = 16;  // 16 * 26 = 416 = RF sample rate / audio sample rate
  static constexpr size_t rfLowPassDecimation = 26;
  static constexpr size_t quadDemodInputSampleRate = audioSampleRate * audioLowPassDecimation;
  static constexpr float rfSampleRate = static_cast<float>(rfLowPassDecimation) * quadDemodInputSampleRate;
  static constexpr float rfLowPassCutoffFrequency = channelWidth;
  static constexpr float rfLowPassTransitionWidth = channelWidth / 2.0f;

  static constexpr float audioLowPassTransitionWidth = audioSampleRate / 2.0f * 0.1f;
  static constexpr float audioLowPassCutoffFrequency = audioSampleRate / 2.0f - audioLowPassTransitionWidth;
};

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

  THROW_IF_ERR(driver->setupNode(deviceToHost, "Copy audio samples from GPU to System Memory"));
  THROW_IF_ERR(driver->setupNode(audioOutput, "Encode and mux audio to a file"));

  *outputByteCountWritten = deviceToHost;

  return ImmutableRef<Sink>(driver);
}

int doAm();
int doFm();

int main() { return doAm(); }

int doAm() {
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

  auto factories = unwrap(getFactoriesSingleton());
  const auto modulation = Modulation_Am;
  const size_t audioSampleRate = 48000;
  const size_t rfLowPassDecim = 4;
  const size_t audioLowPassDecim = 10;
  const float rfSampleRate = audioSampleRate * audioLowPassDecim * rfLowPassDecim;
  const auto tunedCenterFrequency = 1700e3;
  const auto channelFrequency = 1400e3;
  const auto rfLowPassDbAttenuation = -60.0f;
  const auto audioLowPassDbAttenuation = -60.0f;
  const auto channelWidth = kAmChannelBandwidth;
  const auto fskDevationIfFm = 0.0f;
  const int32_t cudaDevice = 0;
  cudaStream_t cudaStream = createCudaStream();
  const char* const outputFileName = "/home/rick/sdr/am.ts";
  const size_t outputBitRate = 128000;

  Ref<Source> inputPipeline;
  if (useFileInput) {
  } else {
    inputPipeline = createHackrfInputPipeline(factories, tunedCenterFrequency, rfSampleRate, cudaDevice, cudaStream);
  }

  Ref<IReadByteCountMonitor> readByteCountMonitor;
  ImmutableRef<Sink> outputPipeline = createAacFileOutputPipeline(
      factories,
      &readByteCountMonitor,
      outputFileName,
      audioSampleRate,
      outputBitRate,
      cudaDevice,
      cudaStream);

  auto rfToAudio = unwrap(factories->getRfToPcmAudioFactory()->create(
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
      cudaStream));

  auto driver = unwrap(factories->getSteppingDriverFactory()->createSteppingDriver());

  THROW_IF_ERR(driver->connect(inputPipeline.get(), 0, rfToAudio, 0));
  THROW_IF_ERR(driver->connect(rfToAudio, 0, outputPipeline, 0));

  THROW_IF_ERR(driver->setupNode(inputPipeline.get(), "RF Input Pipeline"));
  THROW_IF_ERR(driver->setupNode(rfToAudio, "Convert RF signal to audio"));
  THROW_IF_ERR(driver->setupNode(outputPipeline, "Audio Output Pipeline"));

  const size_t maxSampleCount = audioSampleRate * 60;

  size_t itCount = 0;
  while (readByteCountMonitor->getByteCountRead(0) / sizeof(float) < maxSampleCount) {
    THROW_IF_ERR(driver->doFilter());

    /*
    itCount++;

    if (itCount == 100) {
      break;
    }*/
  }

  return 0;
}

int doFm() {
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

  auto factories = unwrap(getFactoriesSingleton());
  const auto modulation = Modulation_Fm;
  const size_t audioSampleRate = 48000;
  const size_t rfLowPassDecim = 26;
  const size_t audioLowPassDecim = 16;
  const float rfSampleRate = audioSampleRate * audioLowPassDecim * rfLowPassDecim;
  const auto tunedCenterFrequency = 97e6;
  const auto channelFrequency = 98.5e6;
  const auto rfLowPassDbAttenuation = -60.0f;
  const auto audioLowPassDbAttenuation = -60.0f;
  const auto channelWidth = kWbfmChannelWidth;
  const auto fskDevationIfFm = kWbfmFrequencyDeviation;
  const int32_t cudaDevice = 0;
  cudaStream_t cudaStream = createCudaStream();
  const char* const outputFileName = "/home/rick/sdr/audio.ts";
  const size_t outputBitRate = 128000;

  Ref<Source> inputPipeline;
  if (useFileInput) {
  } else {
    inputPipeline = createHackrfInputPipeline(factories, tunedCenterFrequency, rfSampleRate, cudaDevice, cudaStream);
  }

  Ref<IReadByteCountMonitor> readByteCountMonitor;
  ImmutableRef<Sink> outputPipeline = createAacFileOutputPipeline(
      factories,
      &readByteCountMonitor,
      outputFileName,
      audioSampleRate,
      outputBitRate,
      cudaDevice,
      cudaStream);

  auto rfToAudio = unwrap(factories->getRfToPcmAudioFactory()->create(
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
      cudaStream));

  auto driver = unwrap(factories->getSteppingDriverFactory()->createSteppingDriver());

  THROW_IF_ERR(driver->connect(inputPipeline.get(), 0, rfToAudio, 0));
  THROW_IF_ERR(driver->connect(rfToAudio, 0, outputPipeline, 0));

  THROW_IF_ERR(driver->setupNode(inputPipeline.get(), "RF Input Pipeline"));
  THROW_IF_ERR(driver->setupNode(rfToAudio, "Convert RF signal to audio"));
  THROW_IF_ERR(driver->setupNode(outputPipeline, "Audio Output Pipeline"));

  const size_t maxSampleCount = audioSampleRate * 60;

  size_t itCount = 0;
  while (readByteCountMonitor->getByteCountRead(0) / sizeof(float) < maxSampleCount) {
    THROW_IF_ERR(driver->doFilter());

    /*
    itCount++;

    if (itCount == 100) {
      break;
    }*/
  }

  return 0;
}
