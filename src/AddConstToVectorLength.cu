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
#include <cuda.h>

#include "AddConstToVectorLength.h"
#include "CudaBuffers.h"
#include "CudaDevicePushPop.h"
#include "cuComplexOperatorOverloads.cuh"
#include "cuda_util.h"

using namespace std;

const size_t AddConstToVectorLength::mAlignment = 32;

__global__ void
k_AddToAmplitude(const cuComplex* in, float addToAmplitude, cuComplex* out) {
  const size_t x = blockDim.x * blockIdx.x + threadIdx.x;

  const cuComplex inputValue = in[x];
  const float oldLength = hypotf(inputValue.x, inputValue.y);
  const cuComplex normVec = inputValue / oldLength;
  const float newLength = addToAmplitude + oldLength;

  out[x] = newLength * normVec;
}

AddConstToVectorLength::AddConstToVectorLength(
    float addValueToAmplitude,
    int32_t cudaDevice,
    cudaStream_t cudaStream)
    : mAddValueToAmplitude(addValueToAmplitude), mCudaDevice(cudaDevice),
      mCudaStream(cudaStream), mBufferCheckedOut(false) {}

shared_ptr<Buffer> AddConstToVectorLength::requestBuffer(
    size_t port,
    size_t numBytes) {
  if (port >= 1) {
    throw runtime_error("Port [" + to_string(port) + "] is out of range");
  }

  if (mBufferCheckedOut) {
    throw runtime_error("Cannot request buffer - it is already checked out");
  }

  CudaDevicePushPop setAndRestore(mCudaDevice);
  ensureMinCapacityAlignedCuda(
      &mInputBuffer,
      numBytes,
      mAlignment * sizeof(cuComplex),
      mCudaStream);

  return mInputBuffer->sliceRemaining();
}

void AddConstToVectorLength::commitBuffer(size_t port, size_t numBytes) {
  if (port >= 1) {
    throw runtime_error("Port [" + to_string(port) + "] is out of range");
  }

  if (!mBufferCheckedOut) {
    throw runtime_error("Buffer cannot be committed - it was not checked out");
  }

  mInputBuffer->increaseEndOffset(numBytes);
  mBufferCheckedOut = false;
}

size_t AddConstToVectorLength::getOutputDataSize(size_t port) {
  return getAvailableNumInputElements() * sizeof(cuComplex);
}

size_t AddConstToVectorLength::getAvailableNumInputElements() const {
  return mInputBuffer->used() / sizeof(cuComplex);
}

size_t AddConstToVectorLength::getOutputSizeAlignment(size_t port) {
  return mAlignment * sizeof(cuComplex);
}

void AddConstToVectorLength::readOutput(
    const vector<shared_ptr<Buffer>>& portOutputs) {
  if (portOutputs.empty()) {
    throw runtime_error("One output port is required");
  }

  CudaDevicePushPop setAndRestore(mCudaDevice);

  const size_t numInputElements = getAvailableNumInputElements();
  const auto& outputBuffer = portOutputs[0];
  const size_t maxNumOutputElements =
      outputBuffer->remaining() / sizeof(cuComplex);

  const size_t maxUnalignedNumElementsToProcess =
      min(numInputElements, maxNumOutputElements);

  const size_t numBlocks = maxUnalignedNumElementsToProcess / mAlignment;
  const size_t processNumInputElements = numBlocks * mAlignment;

  const dim3 blocks = dim3(numBlocks);
  const dim3 threads = dim3(mAlignment);

  k_AddToAmplitude<<<blocks, threads, 0, mCudaStream>>>(
      mInputBuffer->readPtr<cuComplex>(),
      mAddValueToAmplitude,
      outputBuffer->writePtr<cuComplex>());

  const size_t writtenNumBytes = processNumInputElements * sizeof(cuComplex);
  outputBuffer->increaseEndOffset(writtenNumBytes);
  mInputBuffer->increaseOffset(writtenNumBytes);
  moveUsedToStartCuda(mInputBuffer.get(), mCudaStream);
}
