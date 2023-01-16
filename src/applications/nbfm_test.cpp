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
static mutex shutdownMutex;

static constexpr bool useFileInput = true;

static void cleanupThings() {
  while (!runAtExit.empty()) {
    const auto toRun = std::move(runAtExit.top());
    runAtExit.pop();

    toRun();
  }
}

static void exitSigHandler(int signum) {
  printf("Caught signal %d, cleaning up.\n", signum);
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
    THROW("Failed to generate low-pass taps: " << remezStatusToString(status));
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
      : factories(getFactoriesSingleton()),
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
        hackrfSource(factories->getHackrfSourceFactory()->createHackrfSource(0, centerFrequency, rfSampleRate, 3)),
        hostToDevice(
            factories->getCudaMemcpyFilterFactory()->createCudaMemcpy(cudaMemcpyHostToDevice, cudaDevice, cudaStream)),
        convertHackrfInputToFloat(factories->getInt8ToFloatFactory()->createFilter(cudaDevice, cudaStream)),
        cosineSource(factories->getCosineSourceFactory()->createCosineSource(
            rfSampleRate,
            centerFrequency - channelFrequency,
            cudaDevice,
            cudaStream)),
        multiplyRfSourceByCosine(factories->getMultiplyFactory()->createFilter(cudaDevice, cudaStream)),
        rfLowPassFilter(factories->getFirFactory()->createFir(
            FirType_FloatTapsComplexFloatSignal,
            rfLowPassDecimation,
            mRfLowPassTaps.data(),
            mRfLowPassTaps.size(),
            cudaDevice,
            cudaStream)),
        quadDemod(factories->getQuadDemodFactory()
                      ->create(getQuadDemodGain(quadDemodInputSampleRate, channelWidth), cudaDevice, cudaStream)),
        audioLowPassFilter(factories->getFirFactory()->createFir(
            FirType_FloatTapsFloatSignal,
            audioLowPassDecimation,
            mRfLowPassTaps.data(),
            mRfLowPassTaps.size(),
            cudaDevice,
            cudaStream)),
        deviceToHost(
            factories->getCudaMemcpyFilterFactory()->createCudaMemcpy(cudaMemcpyDeviceToHost, cudaDevice, cudaStream)),
        floatQuadReader(factories->getFileReaderFactory()->createFileReader(inputFileName)),
        audioFileWriter(factories->getAacFileWriterFactory()
                            ->createAacFileWriter(outputAudioFile, audioSampleRate, outputAudioBitRate)) {
    printf("Input file [%s]\n", inputFileName);
    printf("Output file [%s] sample rate [%zu] bit rate [%d]\n", outputAudioFile, audioSampleRate, outputAudioBitRate);
    printf("CUDA device [%d] stream [%p]\n", cudaDevice, cudaStream);
    printf(
        "Channel frequency [%f], center frequency [%f] channel width [%f]\n",
        channelFrequency,
        centerFrequency,
        channelWidth);
    printf("RF sample rate [%f]\n", rfSampleRate);
    printf(
        "RF Low-pass cutoff [%f] transition [%f] attenuation [%f] decimation [%zu] tap length [%zu]\n",
        rfLowPassCutoffFrequency,
        rfLowPassTransitionWidth,
        rfLowPassDbAttenuation,
        rfLowPassDecimation,
        mRfLowPassTaps.size());

    printf(
        "Audio Low-pass cutoff [%f] transition [%f] attenuation [%f] decimation [%zu] tap length [%zu]\n",
        audioLowPassCutoffFrequency,
        audioLowPassTransitionWidth,
        audioLowPassDbAttenuation,
        audioLowPassDecimation,
        mAudioLowPassTaps.size());
    printf("Cosine source frequency [%f]\n", centerFrequency - channelFrequency);
    printf("Quad demod gain [%f]\n", getQuadDemodGain(quadDemodInputSampleRate, channelWidth));
  }

  void start() {
    if (!useFileInput) {
      hackrfSource->start();
    }
  }

  void stop() {
    if (!useFileInput) {
      hackrfSource->stop();
    }
  }

  void doFilter(size_t stopAfterOutputSampleCount) {
    size_t wroteSampleCount = 0;

    CudaDevicePushPop deviceSetter(cudaDevice);
    vector<shared_ptr<IBuffer>> outputBuffers(1);

    while (wroteSampleCount < stopAfterOutputSampleCount) {
      Source* gpuFloatSource = nullptr;

      if (useFileInput) {
        shared_ptr<IBuffer> samples = hostToDevice->requestBuffer(0, floatQuadReader->getAlignedOutputDataSize(0));
        outputBuffers[0] = samples;
        floatQuadReader->readOutput(outputBuffers);
        hostToDevice->commitBuffer(0, samples->range()->used());

        if (samples->range()->used() == 0) {
          return;
        }

        gpuFloatSource = hostToDevice.get();
      } else {
        size_t hackrfBufferLength = hackrfSource->getAlignedOutputDataSize(0);
        const size_t hackRfInputSampleCount = hackrfBufferLength / 2;  // samples alternates between real and imaginary

        shared_ptr<IBuffer> hackrfHostSamples = hostToDevice->requestBuffer(0, hackrfBufferLength);

        outputBuffers[0] = hackrfHostSamples;
        hackrfSource->readOutput(outputBuffers);
        hostToDevice->commitBuffer(0, hackrfHostSamples->range()->used());

        shared_ptr<IBuffer> s8ToFloatInputCudaBuffer =
            convertHackrfInputToFloat->requestBuffer(0, hostToDevice->getAlignedOutputDataSize(0));
        outputBuffers[0] = s8ToFloatInputCudaBuffer;
        hostToDevice->readOutput(outputBuffers);

        convertHackrfInputToFloat->commitBuffer(0, s8ToFloatInputCudaBuffer->range()->used());
        gpuFloatSource = convertHackrfInputToFloat.get();
      }

      const size_t multiplyInputBufferSize = gpuFloatSource->getAlignedOutputDataSize(0);
      shared_ptr<IBuffer> rfMultiplyInput =
          multiplyRfSourceByCosine->requestBuffer(0, gpuFloatSource->getAlignedOutputDataSize(0));
      shared_ptr<IBuffer> cosineMultiplyInput =
          multiplyRfSourceByCosine->requestBuffer(1, gpuFloatSource->getAlignedOutputDataSize(0));

      size_t multiplyInputSampleCount =
          min(rfMultiplyInput->range()->remaining() / sizeof(cuComplex),
              cosineMultiplyInput->range()->remaining() / sizeof(cuComplex));

      outputBuffers[0] = rfMultiplyInput;
      gpuFloatSource->readOutput(outputBuffers);

      if (multiplyInputBufferSize != rfMultiplyInput->range()->used()) {
        THROW(
            "Expected source complex-float buffer size of [" << multiplyInputBufferSize << "] but got ["
                                                             << rfMultiplyInput->range()->used() << "]");
      }

      multiplyRfSourceByCosine->commitBuffer(0, rfMultiplyInput->range()->used());

      outputBuffers[0] = cosineMultiplyInput;
      cosineSource->readOutput(outputBuffers);

      multiplyRfSourceByCosine->commitBuffer(1, cosineMultiplyInput->range()->used());

      const shared_ptr<IBuffer> rfLowPassBuffer =
          rfLowPassFilter->requestBuffer(0, multiplyRfSourceByCosine->getAlignedOutputDataSize(0));
      outputBuffers[0] = rfLowPassBuffer;
      multiplyRfSourceByCosine->readOutput(outputBuffers);
      rfLowPassFilter->commitBuffer(0, rfLowPassBuffer->range()->used());

      const shared_ptr<IBuffer> quadDemodInputBuffer =
          quadDemod->requestBuffer(0, rfLowPassFilter->getAlignedOutputDataSize(0));
      outputBuffers[0] = quadDemodInputBuffer;
      rfLowPassFilter->readOutput(outputBuffers);
      quadDemod->commitBuffer(0, quadDemodInputBuffer->range()->used());

      const shared_ptr<IBuffer> audioLowPassBuffer =
          audioLowPassFilter->requestBuffer(0, quadDemod->getAlignedOutputDataSize(0));
      outputBuffers[0] = audioLowPassBuffer;
      quadDemod->readOutput(outputBuffers);
      audioLowPassFilter->commitBuffer(0, audioLowPassBuffer->range()->used());

      const shared_ptr<IBuffer> deviceToHostBuffer =
          deviceToHost->requestBuffer(0, audioLowPassFilter->getAlignedOutputDataSize(0));
      outputBuffers[0] = deviceToHostBuffer;
      audioLowPassFilter->readOutput(outputBuffers);
      deviceToHost->commitBuffer(0, deviceToHostBuffer->range()->used());

      const shared_ptr<IBuffer> audioEncMuxInputBuffer =
          audioFileWriter->requestBuffer(0, deviceToHost->getAlignedOutputDataSize(0));

      outputBuffers[0] = audioEncMuxInputBuffer;
      deviceToHost->readOutput(outputBuffers);

      cudaStreamSynchronize(cudaStream);

      wroteSampleCount += audioEncMuxInputBuffer->range()->used() / sizeof(float);
      audioFileWriter->commitBuffer(0, audioEncMuxInputBuffer->range()->used());
    }
  }

 private:
  const char* const fileName98_5MHz_Fm_Wb = "/home/rick/sdr/98.5MHz.float.iq";
  const char* const fileName145_45MHz_Fm_Nb = "/home/rick/sdr/raw.bin";
  const char* const inputFileName = fileName98_5MHz_Fm_Wb;

  const char* const outputAudioFile = "/home/rick/sdr/allgpu.ts";

  IFactories* const factories;
  const int32_t cudaDevice;
  cudaStream_t cudaStream;
  const vector<float> mRfLowPassTaps;
  const vector<float> mAudioLowPassTaps;
  const shared_ptr<IHackrfSource> hackrfSource;
  const shared_ptr<Filter> hostToDevice;
  const shared_ptr<Filter> convertHackrfInputToFloat;
  const shared_ptr<Source> cosineSource;
  const shared_ptr<Filter> multiplyRfSourceByCosine;
  const shared_ptr<Filter> rfLowPassFilter;
  const shared_ptr<Filter> quadDemod;
  const shared_ptr<Filter> audioLowPassFilter;
  const shared_ptr<Filter> deviceToHost;
  const shared_ptr<Source> floatQuadReader;
  const shared_ptr<Sink> audioFileWriter;

  /*
  static constexpr float maxRfSampleRate = 20e6;
  static constexpr float audioSampleRate = 48e3;
  static constexpr int32_t outputAudioBitRate = 128000;
  static constexpr float centerFrequency = 144e6;
  static constexpr float channelFrequency = 145.45e6;
  static constexpr float rfLowPassDbAttenuation = -60.0f;
  static constexpr float lowPassGain = 1.0f;
*/

  static constexpr float maxRfSampleRate = 20e6;
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
  static constexpr float audioLowPassTransitionWidth = audioSampleRate / 2.0f * 0.1f;
  static constexpr float audioLowPassCutoffFrequency = audioSampleRate / 2.0f - audioLowPassTransitionWidth;
};

int main(int argc, char** argv) {
  atexit(cleanupThings);
  signal(SIGSEGV, &exitSigHandler);
  signal(SIGINT, &exitSigHandler);
  signal(SIGTERM, &exitSigHandler);

  const size_t stopAfterOutputAudioSampleCount = 48000 * 60;

  auto nbfmTest = NbfmTest();
  nbfmTest.start();
  nbfmTest.doFilter(stopAfterOutputAudioSampleCount);
  nbfmTest.stop();
}
