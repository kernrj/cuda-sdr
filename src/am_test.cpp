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

#include <cstdint>

#include "AddConst.h"
#include "CosineSource.h"
#include "HackrfSource.h"
#include "Magnitude.h"
#include "Multiply.h"
#include "S8ToFloat.h"
#include "cuda_util.h"
#include "fir.h"
#include "remez.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

using namespace std;

int main(int argc, char** argv) {
  const int32_t cudaDevice = 0;
  const float maxRfSampleRate = 20e6;
  const float audioSampleRate = 48e3;
  const auto decimation =
      static_cast<size_t>((floorf(maxRfSampleRate / audioSampleRate)));
  const float rfSampleRate = static_cast<float>(decimation) * audioSampleRate;
  const float centerFrequency = 1350e3;
  const float channelFrequency = 1340e3;
  const float channelPassBandWidth = 9.69e3;
  const float channelEndBandWidth = 10.15e3;
  const float cutoffFrequency = channelPassBandWidth / 2.0f;
  const float transitionWidth =
      (channelEndBandWidth - channelPassBandWidth) / 2.0f;

  cudaStream_t cudaStream = nullptr;
  SAFE_CUDA(cudaStreamCreate(&cudaStream));
  HackrfSource hackrfSource;
  hackrfSource.selectDeviceByIndex(0);

  CudaInt8ToFloat convertHackrfInputToFloat(cudaDevice, cudaStream);

  CosineSource cosineSource(
      rfSampleRate,
      centerFrequency - channelFrequency,
      cudaDevice,
      cudaStream);

  MultiplyCcc multiplyRfSourceByCosine(cudaDevice, cudaStream);

  const float gain = 1.0f;

  std::vector<float> lowPassTaps =
      generateLowPassTaps(gain, rfSampleRate, cutoffFrequency, transitionWidth);

  FirCcf lowPassFilter(decimation, lowPassTaps, cudaDevice, cudaStream);
  Magnitude magnitude(cudaDevice, cudaStream);
  AddConst centerAtZero(-1.0f, cudaDevice, cudaStream);

  int8_t* hackrfHostSamples = nullptr;
  hackrfSource.setSampleCallback(
      [&hackrfHostSamples,
       &convertHackrfInputToFloat,
       &cosineSource,
       &multiplyRfSourceByCosine](const int8_t* samples, size_t byteCount) {
        const size_t hackRfInputSampleCount =
            byteCount / 2;  // samples alternates between real and imaginary
        shared_ptr<Buffer> s8ToFloatInputCudaBuffer =
            convertHackrfInputToFloat.requestBuffer(0, byteCount);
        if (s8ToFloatInputCudaBuffer->remaining() < byteCount) {
          printf("Dropped\n");
        }

        size_t copyByteCount =
            min(byteCount, s8ToFloatInputCudaBuffer->remaining());
        memmove(hackrfHostSamples, samples, copyByteCount);
        cudaMemcpyAsync(
            s8ToFloatInputCudaBuffer->writePtr<int8_t>(),
            hackrfHostSamples,
            copyByteCount,
            cudaMemcpyHostToDevice);
        memmove(
            s8ToFloatInputCudaBuffer->writePtr<int8_t>(),
            samples,
            copyByteCount);
        convertHackrfInputToFloat.commitBuffer(0, copyByteCount);

        const size_t multiplyInputCount =
            hackRfInputSampleCount * sizeof(cuComplex);
        shared_ptr<Buffer> rfMultiplyInput =
            multiplyRfSourceByCosine.requestBuffer(0, multiplyInputCount);
        shared_ptr<Buffer> cosineMultiplyInput =
            multiplyRfSourceByCosine.requestBuffer(1, multiplyInputCount);

        size_t multiplyInputSampleCount =
            min(rfMultiplyInput->remaining() / sizeof(cuComplex),
                cosineMultiplyInput->remaining() / sizeof(cuComplex));
      });
}
