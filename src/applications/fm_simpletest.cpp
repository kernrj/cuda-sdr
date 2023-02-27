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
#include <gpusdrpipeline/Factories.h>
#include <gpusdrpipeline/GSLog.h>
#include <gpusdrpipeline/buffers/IBuffer.h>
#include <gpusdrpipeline/filters/Filter.h>
#include <gpusdrpipeline/filters/IHackrfSource.h>
#include <gpusdrpipeline/fm.h>
#include <gpusdrpipeline/util/CudaDevicePushPop.h>
#include <gsdr/gsdr.h>
#include <libhackrf/hackrf.h>
#include <remez/remez.h>

#include <atomic>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stack>

using namespace std;

// Example: SSTREAM("hello " << username << " you are " << age << " years old.");
#define SSTREAM(x) dynamic_cast<std::ostringstream&&>(std::ostringstream() << x).str()

// Example: GS_FAIL("Illegal argument: " << argument);
#define GS_FAIL(x)                   \
  do {                               \
    auto __msg = SSTREAM(x);         \
    std::cerr << __msg << std::endl; \
    throw std::runtime_error(__msg); \
  } while (false)

#define SAFERF(__cmd, __errorMsg)                                                    \
  do {                                                                               \
    int status = (__cmd);                                                            \
    if (status != HACKRF_SUCCESS) {                                                  \
      const string errorName = hackrf_error_name(static_cast<hackrf_error>(status)); \
      GS_FAIL(__errorMsg << " - Error " << errorName << " (" << status << ')');      \
    }                                                                                \
  } while (false)

#ifdef DEBUG
#define GPU_SYNC_DEBUG cudaDeviceSynchronize()
#else
#define GPU_SYNC_DEBUG cudaSuccess
#endif

#ifdef DEBUG
#define CHECK_CUDA(msgOnFail__)                                                                                 \
  do {                                                                                                          \
    cudaError_t checkCudaStatus__ = cudaDeviceSynchronize();                                                    \
    if (checkCudaStatus__ != cudaSuccess) {                                                                     \
      GS_FAIL(                                                                                                  \
          (msgOnFail__) << ": " << cudaGetErrorName(checkCudaStatus__) << " (" << checkCudaStatus__ << "). At " \
                        << __FILE__ << ':' << __LINE__);                                                        \
    }                                                                                                           \
  } while (false)
#else
#define CHECK_CUDA(__msg) (void)0
#endif

#define SAFE_CUDA(cudaCmd__)                                                                                  \
  do {                                                                                                        \
    CHECK_CUDA("Before: " #cudaCmd__);                                                                        \
    cudaError_t safeCudaStatus__ = (cudaCmd__);                                                               \
    if (safeCudaStatus__ != cudaSuccess) {                                                                    \
      GS_FAIL(                                                                                                \
          "CUDA error " << cudaGetErrorName(safeCudaStatus__) << ": " << cudaGetErrorString(safeCudaStatus__) \
                        << ". At " << __FILE__ << ':' << __LINE__);                                           \
    }                                                                                                         \
    CHECK_CUDA("After: " #cudaCmd__);                                                                         \
  } while (false)

static stack<function<void()>> runAtExit;
static atomic_bool wasInterrupted(false);
static condition_variable wasInterruptedCv;

static constexpr bool useFileInput = false;

static void cleanupThings() {
  if (runAtExit.empty()) {
    gslogi("Nothing to clean up");
  } else {
    gslogi("Cleaning up...");
  }

  while (!runAtExit.empty()) {
    const auto toRun = std::move(runAtExit.top());
    runAtExit.pop();

    toRun();
  }
}

static void exitSigHandler(int signum) {
  gslogi("Caught signal %d, cleaning up.", signum);
  cleanupThings();

  if (signum == SIGINT || signum == SIGTERM) {
    wasInterrupted.store(true);
    wasInterruptedCv.notify_all();
  } else {
    abort();
  }
}

static cudaStream_t createCudaStream() {
  cudaStream_t cudaStream = nullptr;
  SAFE_CUDA(cudaStreamCreate(&cudaStream));

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
        quadDemod(unwrap(
            factories->getQuadDemodFactory()
                ->create(Modulation_Fm, quadDemodInputSampleRate, kWbfmFrequencyDeviation, cudaDevice, cudaStream))),
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
    gslogi(
        "Output file [%s] sample rate [%zu] bit rate [%d]",
        outputAudioFile,
        audioSampleRate,
        outputAudioBitRate);
    gslogi("CUDA device [%d] stream [%p]", cudaDevice, cudaStream);
    gslogi(
        "Channel frequency [%f], center frequency [%f] channel width [%f]",
        channelFrequency,
        centerFrequency,
        channelWidth);
    gslogi("RF sample rate [%zu]", rfSampleRate);
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
      hackrfSource->start();
    }
  }

  void stop() {
    if (!useFileInput) {
      gslogi("stop() - live");
      hackrfSource->stop();
      hackrfSource->releaseDevice();
    } else {
      gslogi("stop() - file");
    }
  }

  Status doFilter(size_t stopAfterOutputSampleCount) {
    size_t receivedRfSampleCount = 0;
    size_t wroteSampleCount = 0;

    CUDA_DEV_PUSH_POP_OR_RET_STATUS(cudaDevice);
    vector<IBuffer*> outputBuffers(1);

    auto cudaAllocator =
        unwrap(factories->getCudaAllocatorFactory()->createCudaAllocator(cudaDevice, cudaStream, 32, false));
    auto hostToDeviceMemcpy = unwrap(
        factories->getCudaBufferCopierFactory()->createBufferCopier(cudaDevice, cudaStream, cudaMemcpyHostToDevice));
    auto deviceToDeviceMemcpy = unwrap(
        factories->getCudaBufferCopierFactory()->createBufferCopier(cudaDevice, cudaStream, cudaMemcpyDeviceToDevice));
    auto cudaBufferFactory =
        unwrap(factories->createRelocatableResizableBufferFactory(cudaAllocator, deviceToDeviceMemcpy));

    ImmutableRef<IRelocatableResizableBuffer> rfOutputBuffer = unwrap(cudaBufferFactory->createRelocatableBuffer(1 << 20));
    const size_t lowPassTapCount = mRfLowPassTaps.size();
    const size_t lowPassBufferSize = lowPassTapCount * sizeof(float);
    const auto gpuRfLowPassTaps = unwrap(cudaAllocator->allocate(lowPassBufferSize));
    THROW_IF_ERR(hostToDeviceMemcpy->copy(gpuRfLowPassTaps->data(), mRfLowPassTaps.data(), lowPassBufferSize));

    while (wroteSampleCount < stopAfterOutputSampleCount) {
      Source* gpuFloatSource = nullptr;

      if (useFileInput) {
        ConstRef<IBuffer> samples = unwrap(hostToDevice->requestBuffer(0, floatQuadReader->getAlignedOutputDataSize(0)));

        outputBuffers[0] = samples;
        THROW_IF_ERR(floatQuadReader->readOutput(outputBuffers.data(), 1));
        THROW_IF_ERR(hostToDevice->commitBuffer(0, samples->range()->used()));

        gslogi("Read [%zu] bytes from [%s]", samples->range()->used(), inputFileName);
        if (samples->range()->used() == 0) {
          return Status_Success;
        }

        gpuFloatSource = hostToDevice.get();
      } else {
        size_t hackrfBufferLength = hackrfSource->getAlignedOutputDataSize(0);
        const size_t hackRfInputSampleCount = hackrfBufferLength / 2;  // samples alternates between real and imaginary

        ConstRef<IBuffer> hackrfHostSamples = unwrap(hostToDevice->requestBuffer(0, hackrfBufferLength));

        outputBuffers[0] = hackrfHostSamples;
        THROW_IF_ERR(hackrfSource->readOutput(outputBuffers.data(), 1));
        THROW_IF_ERR(hostToDevice->commitBuffer(0, hackrfHostSamples->range()->used()));

        gslogi("Read [%zu] bytes from HackRF", hackrfHostSamples->range()->used());

        ConstRef<IBuffer> s8ToFloatInputCudaBuffer =
            unwrap(convertHackrfInputToFloat->requestBuffer(0, hostToDevice->getAlignedOutputDataSize(0)));
        outputBuffers[0] = s8ToFloatInputCudaBuffer;
        THROW_IF_ERR(hostToDevice->readOutput(outputBuffers.data(), 1));

        gslogi("Read [%zu] bytes from Int8->Float", s8ToFloatInputCudaBuffer->range()->used());

        THROW_IF_ERR(convertHackrfInputToFloat->commitBuffer(0, s8ToFloatInputCudaBuffer->range()->used()));
        gpuFloatSource = convertHackrfInputToFloat.get();

        gslogi(
            GSLOG_INFO,
            "Available [%zu] [%zu]",
            convertHackrfInputToFloat->getOutputDataSize(0),
            convertHackrfInputToFloat->getAlignedOutputDataSize(0));
      }

      size_t firstSampleOffset = receivedRfSampleCount % rfSampleRate;
      size_t rfDataSize = gpuFloatSource->getOutputDataSize(0);
      size_t inputSampleCount = rfDataSize / sizeof(cuComplex);
      receivedRfSampleCount += inputSampleCount;

      size_t alignedSourceBufferSize = gpuFloatSource->getAlignedOutputDataSize(0);

      gslogi("%zu %zu %zu", rfDataSize, inputSampleCount, alignedSourceBufferSize);
      THROW_IF_ERR(rfOutputBuffer->ensureMinSize(alignedSourceBufferSize));
      outputBuffers[0] = rfOutputBuffer;
      THROW_IF_ERR(gpuFloatSource->readOutput(outputBuffers.data(), 1));

      if (inputSampleCount / rfLowPassDecimation <= 1) {
        continue;  // need two low-pass outputs to demodulate one FM sample.
      }

      static_assert(sizeof(cuComplex) == 8, "Make sure FM demod alignment still works");
      const size_t fmDemodOutputCount = (inputSampleCount / rfLowPassDecimation - 1) & -(32 / sizeof(cuComplex));

      ConstRef<IBuffer> audioLowPassBuffer = unwrap(audioLowPassFilter->requestBuffer(0, fmDemodOutputCount));

      gslogi(
          "%p %p %p %p",
          gpuRfLowPassTaps.get(),
          rfOutputBuffer->readPtr<cuComplex>(),
          audioLowPassBuffer->writePtr<float>(),
          audioLowPassBuffer->base());
      SAFE_CUDA(gsdrFmDemod(
          rfSampleRate,
          centerFrequency,
          channelFrequency,
          channelWidth,
          channelFmDeviation,
          rfLowPassDecimation,
          firstSampleOffset,
          gpuRfLowPassTaps->as<float>(),
          lowPassTapCount,
          rfOutputBuffer->readPtr<cuComplex>(),
          audioLowPassBuffer->writePtr<float>(),
          fmDemodOutputCount,
          cudaDevice,
          cudaStream));

      THROW_IF_ERR(rfOutputBuffer->relocateUsedToStart());

      THROW_IF_ERR(audioLowPassFilter->commitBuffer(0, fmDemodOutputCount));

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

  // 98.5MHz channel (WBFM) file capture. RF sample rate 4.8MHz, center frequency 97.5MHz.
  static constexpr size_t audioSampleRate = 48e3;
  static constexpr int32_t outputAudioBitRate = 128000;
  static constexpr float centerFrequency = 97.5e6;
  static constexpr float channelFrequency = 98.5e6;
  static constexpr float channelFmDeviation = kWbfmFrequencyDeviation;
  static constexpr float rfLowPassDbAttenuation = -60.0f;
  static constexpr float audioLowPassDbAttenuation = -60.0f;
  static constexpr float channelWidth = kWbfmChannelWidth;
  static constexpr size_t audioLowPassDecimation = 10;
  static constexpr size_t rfLowPassDecimation = 10;
  static constexpr size_t quadDemodInputSampleRate = audioSampleRate * audioLowPassDecimation;
  static constexpr size_t rfSampleRate = static_cast<float>(rfLowPassDecimation) * quadDemodInputSampleRate;
  static constexpr float rfLowPassCutoffFrequency = channelWidth;
  static constexpr float rfLowPassTransitionWidth = channelWidth / 2.0f;

  /*
    // 98.5MHz channel (WBFM) live.
    static constexpr size_t maxRfSampleRate = 19.968e6;  // Largest multiple of 48KHz less than 20MHz, 416 * 48000
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
    static constexpr size_t rfSampleRate = rfLowPassDecimation * quadDemodInputSampleRate;
    static constexpr float rfLowPassCutoffFrequency = channelWidth;
    static constexpr float rfLowPassTransitionWidth = channelWidth / 2.0f;
  */
  static constexpr float audioLowPassTransitionWidth = audioSampleRate / 2.0f * 0.1f;
  static constexpr float audioLowPassCutoffFrequency = audioSampleRate / 2.0f - audioLowPassTransitionWidth;
};

int main(int argc, char** argv) {
  atexit(cleanupThings);
  signal(SIGSEGV, &exitSigHandler);
  signal(SIGINT, &exitSigHandler);
  signal(SIGTERM, &exitSigHandler);

  shared_ptr<NbfmTest> nbfmTest;
  try {
    const size_t stopAfterOutputAudioSampleCount = 48000 * 60;
    const float frequency = 97.0e6;
    const float sampleRate = 19968000.0f;

    nbfmTest = make_shared<NbfmTest>();
    nbfmTest->start();
    nbfmTest->doFilter(stopAfterOutputAudioSampleCount);
  } catch (exception& e) {
    gsloge("Exception: %s", e.what());
    nbfmTest->stop();
    nbfmTest.reset();
    gsloge("Done cleaning up");

    return -1;
  } catch (...) {
    gsloge("Unknown error");
    cleanupThings();
    nbfmTest->stop();
    nbfmTest.reset();
    gsloge("Done cleaning up");

    return -1;
  }

  return 0;
}
