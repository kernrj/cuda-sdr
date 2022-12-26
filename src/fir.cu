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
#include <cuda_runtime_api.h>

#include <string>

#include "CudaBuffers.h"
#include "CudaDevicePushPop.h"
#include "ScopeExit.h"
#include "cuComplexOperatorOverloads.cuh"
#include "cuda_util.h"
#include "fir.h"
#include "remez.h"

using namespace std;

#include <complex>

const size_t FirCcf::mAlignment = 32;

template <class IN_T, class OUT_T, class TAP_T>
__global__ void k_Fir(const IN_T* input, const TAP_T* tapsReversed, uint32_t numTaps, OUT_T* output) {
  uint32_t outputIndex = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t inputIndex = outputIndex;

  OUT_T& outputSample = output[outputIndex];
  const IN_T* inputSample = input + inputIndex;

  outputSample = zero<OUT_T>();
  for (uint32_t i = 0; i < numTaps; i++, inputSample++) {
    outputSample += *inputSample * tapsReversed[i];
  }
}

template <class IN_T, class OUT_T, class TAP_T>
__global__ void k_FirDecimate(
    const IN_T* __restrict__ input,
    const TAP_T* __restrict__ tapsReversed,
    uint32_t numTaps,
    uint32_t decimation,
    OUT_T* __restrict__ output) {
  uint32_t outputIndex = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t inputIndex = decimation * outputIndex;

  OUT_T& outputSample = output[outputIndex];
  const IN_T* inputSample = input + inputIndex;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
  // printf("outputIndex [%u] inputIndex [%u]\n", outputIndex, inputIndex);
#endif

  outputSample = zero<OUT_T>();
  for (uint32_t i = 0; i < numTaps; i++, inputSample++) {
    outputSample += *inputSample * tapsReversed[i];
  }
}

FirCcf::FirCcf(size_t decimation, const std::vector<float>& taps, int32_t cudaDevice, cudaStream_t cudaStream)
    : mDecimation(decimation), mTapCount(0), mCudaStream(cudaStream), mCudaDevice(cudaDevice),
      mBufferCheckedOut(false) {
  setTaps(taps);
}

void FirCcf::setTaps(const std::vector<float>& tapsReversed) {
  CudaDevicePushPop setAndRestore(mCudaDevice);

  if (mTapCount < tapsReversed.size()) {
    mTaps = createAlignedBufferCuda(tapsReversed.size() * sizeof(float), 1, mCudaStream);
  }

  SAFE_CUDA(cudaMemcpyAsync(
      mTaps->writePtr(),
      tapsReversed.data(),
      tapsReversed.size() * sizeof(float),
      cudaMemcpyHostToDevice,
      mCudaStream));

  mTapCount = static_cast<int32_t>(tapsReversed.size());
}

shared_ptr<Buffer> FirCcf::requestBuffer(size_t port, size_t numBytes) {
  if (mBufferCheckedOut) {
    throw runtime_error("Cannot request buffer - it is already checked out");
  }

  const size_t endOffset = mInputBuffer != nullptr ? mInputBuffer->endOffset() : 0;
  ensureMinCapacityAlignedCuda(&mInputBuffer, endOffset + numBytes, mAlignment * sizeof(cuComplex), mCudaStream);

  mBufferCheckedOut = true;
  const auto buffer = mInputBuffer->sliceRemaining();
  buffer->reset();

  return buffer;
}

void FirCcf::commitBuffer(size_t port, size_t numBytes) {
  if (!mBufferCheckedOut) {
    throw runtime_error("Cannot commit mInputBuffer - it was not checked out");
  }

  if (port > 0) {
    throw runtime_error("Only 1 input port is supported");
  }

  mInputBuffer->increaseEndOffset(numBytes);
  mBufferCheckedOut = false;
}

size_t FirCcf::getOutputDataSize(size_t port) {
  if (port != 0) {
    throw runtime_error("Output port [" + to_string(port) + "] is out of range");
  }

  const size_t numInputElements = mInputBuffer->used() / sizeof(cuComplex);
  const size_t numOutputElements = numInputElements / mDecimation;

  const size_t outputDataSize = numOutputElements * sizeof(cuComplex);

  return outputDataSize;
}

size_t FirCcf::getOutputSizeAlignment(size_t port) {
  if (port != 0) {
    throw runtime_error("Output port [" + to_string(port) + "] is out of range");
  }

  return mAlignment * sizeof(cuComplex);
}

void FirCcf::readOutput(const vector<shared_ptr<Buffer>>& portOutputs) {
  if (portOutputs.empty()) {
    throw runtime_error("Must have one output port");
  }

  if (mInputBuffer->used() < mTapCount) {
    return;
  }

  CudaDevicePushPop setAndRestore(mCudaDevice);

  const auto& outputBuffer = portOutputs[0];

  const size_t maxOutputsInOutputBuffer = outputBuffer->remaining() / sizeof(cuComplex);
  const size_t maxBlocksInOutputBuffer = maxOutputsInOutputBuffer / mAlignment;
  const size_t availableNumOutputs = getOutputDataSize(0) / sizeof(cuComplex);
  const size_t numAvailableBlocks = availableNumOutputs / mAlignment;
  const size_t numBlocks = min(maxBlocksInOutputBuffer, numAvailableBlocks);

  if (numBlocks == 0) {
    return;
  }

  const dim3 blocks = dim3(numBlocks);
  const dim3 threads = dim3(mAlignment);

  if (mDecimation <= 1) {
    k_Fir<<<blocks, threads, 0, mCudaStream>>>(
        mInputBuffer->readPtr<cuComplex>(),
        mTaps->readPtr<float>(),
        mTapCount,
        outputBuffer->writePtr<cuComplex>());
  } else {
    checkCuda("Before k_FirDecimate()");
    k_FirDecimate<<<blocks, threads, 0, mCudaStream>>>(
        mInputBuffer->readPtr<cuComplex>(),
        mTaps->readPtr<float>(),
        mTapCount,
        mDecimation,
        outputBuffer->writePtr<cuComplex>());
    checkCuda("After k_FirDecimate()");
  }

  const size_t numOutputs = min(numBlocks * mAlignment, availableNumOutputs);
  const size_t numOutputBytes = numOutputs * sizeof(cuComplex);
  outputBuffer->increaseEndOffset(numOutputBytes);

  const size_t numInputElementsUsed = numOutputs * mDecimation;
  const size_t numInputBytesUsed = numInputElementsUsed * sizeof(cuComplex);
  mInputBuffer->increaseOffset(numInputBytesUsed);
  moveUsedToStartCuda(mInputBuffer.get(), mCudaStream);
}

vector<float>
generateLowPassTaps(float sampleFrequency, float cutoffFrequency, float transitionWidth, float dbAttenuation) {
  // kaiserWindowLength(dbAttenuation, transitionWidth)
  //  remez() takes normalized frequencies where 0.5 is the Nyquist rate.
  const float nyquistFrequency = sampleFrequency / 2.0f;
  const float relativeCutoffFrequency = cutoffFrequency / nyquistFrequency;
  const float relativeTransitionWidth = transitionWidth / nyquistFrequency;

  vector<Band> bands {
      Band {
          .lowFrequency = 0.0f,
          .highFrequency = relativeCutoffFrequency,
          .lowFrequencyResponse = 1.0f,
          .highFrequencyResponse = 1.0f,
          .weight = 1.0f},
      Band {
          .lowFrequency = relativeCutoffFrequency + relativeTransitionWidth,
          .highFrequency = 1.0f,
          .lowFrequencyResponse = 0.0f,
          .highFrequencyResponse = 0.0f,
          .weight = 10.0f}};

  vector<double> taps(1024);
  /*
  const FilterType type = Bandpass;
  const size_t gridDensity = 32;
  const size_t maxIterations = 1000;

  RemezStatus status;
  do {
    status = remezNew(taps, bands, type, gridDensity, maxIterations);

    if (status == TooManyExtremalFrequencies) {
      taps.resize(taps.size() * 2);
      printf("Resizing taps to %zd\n", taps.size());
    }
  } while (status == TooManyExtremalFrequencies);

  if (status != Ok) {
    throw runtime_error(
        "Could not calculate low-pass filter taps: "
        + remezStatusToString(status));
  }
  */

  vector<double> bandFrequencies;
  vector<double> bandResponses;
  vector<double> bandEdgeWeights;
  for (const auto& band : bands) {
    bandFrequencies.push_back(band.lowFrequency);
    bandFrequencies.push_back(band.highFrequency);
    bandResponses.push_back(band.lowFrequencyResponse);
    bandResponses.push_back(band.highFrequencyResponse);
    bandEdgeWeights.push_back(band.weight);
  }

  int numTaps = taps.size();
  int numBands = bands.size();
  int type = BANDPASS;
  int gridDensity = 16;

  remez(
      taps.data(),
      &numTaps,
      &numBands,
      bandFrequencies.data(),
      bandResponses.data(),
      bandEdgeWeights.data(),
      &type,
      &gridDensity);

  vector<float> floatTaps(taps.size());

  int r = taps.size() / 2 + 1;
  double delf = 0.5 / gridDensity * r;

  for (size_t i = 0; i < floatTaps.size(); i++) {
    floatTaps[i] = static_cast<float>(taps[i]);
  }

  return floatTaps;
}

/**
 * https://tomroelandts.com/articles/how-to-create-a-configurable-filter-using-a-kaiser-window
 *
 * @param dbAttenuation
 * @param transitionWidthNormalized
 * @return
 */
size_t kaiserWindowLength(float dbAttenuation, float transitionWidthNormalized) {
  const size_t windowLength =
      lrintf(ceilf((dbAttenuation - 8.0f) / (2.285f * 2.0f * M_PIf * transitionWidthNormalized) + 1))
      | 1;  // | 1 to make it odd if even.

  return windowLength;
}
