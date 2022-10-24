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

using namespace std;

#include <complex>

const size_t FirCcf::mAlignment = 32;

template <class IN_T, class OUT_T, class TAP_T>
__global__ void k_Fir(
    const IN_T* input,
    const TAP_T* tapsReversed,
    uint32_t numTaps,
    OUT_T* output) {
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

  outputSample = zero<OUT_T>();
  for (uint32_t i = 0; i < numTaps; i++, inputSample++) {
    outputSample += *inputSample * tapsReversed[i];
  }
}

FirCcf::FirCcf(
    uint32_t decimation,
    const std::vector<float>& taps,
    int32_t cudaDevice,
    cudaStream_t cudaStream)
    : mDecimation(decimation), mTapCount(0), mCudaStream(cudaStream),
      mCudaDevice(cudaDevice), mBufferCheckedOut(false) {
  setTaps(taps);
}

void FirCcf::setTaps(const std::vector<float>& tapsReversed) {
  CudaDevicePushPop setAndRestore(mCudaDevice);

  if (mTapCount < tapsReversed.size()) {
    mTaps = createAlignedBuffer(
        tapsReversed.size() * sizeof(float),
        1,
        mCudaStream);
  }

  SAFE_CUDA(cudaMemcpy(
      mTaps.writePtr(),
      tapsReversed.data(),
      tapsReversed.size() * sizeof(float),
      cudaMemcpyHostToDevice));

  mTapCount = static_cast<int32_t>(tapsReversed.size());
}

Buffer FirCcf::requestBuffer(size_t port, size_t numBytes) {
  ensureMinCapacityAligned(
      &mBuffer,
      mBuffer.remaining() + numBytes,
      mAlignment * sizeof(complex<float>),
      mCudaStream);

  if (mBufferCheckedOut) {
    throw runtime_error("Cannot request buffer - it is already checked out");
  }

  mBufferCheckedOut = true;
  return mBuffer.sliceRemainingUnowned();
}

void FirCcf::commitBuffer(size_t port, size_t numBytes) {
  if (!mBufferCheckedOut) {
    throw runtime_error("Cannot commit buffer - it was not checked out");
  }

  if (mBuffer.end + numBytes > mBuffer.capacity) {
    throw runtime_error(
        "Unable to commit [" + to_string(numBytes) + "] bytes. The maximum is ["
        + to_string(mBuffer.capacity - mBuffer.end) + "]");
  }

  mBufferCheckedOut = false;
  mBuffer.end += numBytes;
}

size_t FirCcf::getOutputDataSize(size_t port) {
  if (port != 0) {
    throw runtime_error(
        "Output port [" + to_string(port) + "] is out of range");
  }

  return mBuffer.remaining() / sizeof(complex<float>) / mDecimation;
}

size_t FirCcf::getOutputSizeAlignment(size_t port) {
  if (port != 0) {
    throw runtime_error(
        "Output port [" + to_string(port) + "] is out of range");
  }

  return mAlignment * sizeof(cuComplex);
}

void FirCcf::readOutput(Buffer* portOutputs, size_t portOutputCount) {
  if (portOutputCount < 1) {
    throw runtime_error("Must have one output port");
  }

  if (mBuffer.remaining() < mTapCount) {
    return;
  }

  CudaDevicePushPop setAndRestore(mCudaDevice);

  Buffer& outputBuffer = portOutputs[0];

  const size_t maxOutputsInOutputBuffer =
      outputBuffer.remaining() / sizeof(cuComplex);
  const size_t maxBlocksInOutputBuffer = maxOutputsInOutputBuffer / mAlignment;
  const size_t availableNumOutputs = getOutputDataSize(0);
  const size_t blockCountToCoverAvailableData =
      (availableNumOutputs + mAlignment - 1) / mAlignment;
  const size_t numBlocks =
      min(blockCountToCoverAvailableData, maxBlocksInOutputBuffer);

  if (numBlocks == 0) {
    return;
  }

  const dim3 blocks = dim3(numBlocks);
  const dim3 threads = dim3(mAlignment);

  if (mDecimation <= 1) {
    k_Fir<<<blocks, threads, 0, mCudaStream>>>(
        mBuffer.readPtr<cuComplex>(),
        mTaps.readPtr<float>(),
        mTapCount,
        outputBuffer.writePtr<cuComplex>());
  } else {
    k_FirDecimate<<<blocks, threads, 0, mCudaStream>>>(
        mBuffer.readPtr<cuComplex>(),
        mTaps.readPtr<float>(),
        mTapCount,
        mDecimation,
        outputBuffer.writePtr<cuComplex>());
  }

  const size_t remainingNumInputBytes = mBuffer.remaining();

  if (mBuffer.hasRemaining()) {
    SAFE_CUDA(cudaMemcpyAsync(
        mBuffer.buffer.get(),
        mBuffer.readPtr(),
        remainingNumInputBytes,
        cudaMemcpyDeviceToDevice,
        mCudaStream));
  }

  const size_t numOutputs = min(numBlocks * mAlignment, availableNumOutputs);
  const size_t numOutputBytes = numOutputs * sizeof(cuComplex);
  outputBuffer.end += numOutputBytes;

  mBuffer.offset = 0;
  mBuffer.end = remainingNumInputBytes;
}
