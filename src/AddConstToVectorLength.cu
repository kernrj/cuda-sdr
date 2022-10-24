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
      mCudaStream(cudaStream) {}

Buffer AddConstToVectorLength::requestBuffer(size_t port, size_t numBytes) {
  if (port >= 1) {
    throw runtime_error("Port [" + to_string(port) + "] is out of range");
  }

  CudaDevicePushPop setAndRestore(mCudaDevice);
  ensureMinCapacityAligned(
      &mInputBuffer,
      numBytes,
      mAlignment * sizeof(cuComplex),
      mCudaStream);

  return mInputBuffer.sliceRemainingUnowned();
}

void AddConstToVectorLength::commitBuffer(size_t port, size_t numBytes) {
  if (port >= 1) {
    throw runtime_error("Port [" + to_string(port) + "] is out of range");
  }

  OwnedBuffer& buffer = mInputBuffer;

  const size_t newEndIndex = buffer.end + numBytes;

  if (newEndIndex > buffer.capacity) {
    throw runtime_error(
        "Committed byte count [" + to_string(numBytes) + "] at offset ["
        + to_string(buffer.end) + "] exceeds capacity ["
        + to_string(buffer.capacity) + "]");
  }

  buffer.end += numBytes;
}

size_t AddConstToVectorLength::getOutputDataSize(size_t port) {
  return getAvailableNumInputElements() * sizeof(cuComplex);
}

size_t AddConstToVectorLength::getAvailableNumInputElements() const {
  return mInputBuffer.used() / sizeof(cuComplex);
}

size_t AddConstToVectorLength::getOutputSizeAlignment(size_t port) {
  return mAlignment * sizeof(cuComplex);
}

void AddConstToVectorLength::readOutput(
    Buffer* portOutputs,
    size_t portOutputCount) {
  if (portOutputCount < 1) {
    throw runtime_error("One output port is required");
  }

  CudaDevicePushPop setAndRestore(mCudaDevice);

  const size_t numInputElements = getAvailableNumInputElements();
  Buffer& outputBuffer = portOutputs[0];
  const size_t maxNumOutputElements =
      outputBuffer.remaining() / sizeof(cuComplex);

  const size_t maxUnalignedNumElementsToProcess =
      min(numInputElements, maxNumOutputElements);

  const size_t numBlocks = maxUnalignedNumElementsToProcess / mAlignment;
  const size_t processNumInputElements = numBlocks * mAlignment;

  const dim3 blocks = dim3(numBlocks);
  const dim3 threads = dim3(mAlignment);

  k_AddToAmplitude<<<blocks, threads, 0, mCudaStream>>>(
      mInputBuffer.readPtr<cuComplex>(),
      mAddValueToAmplitude,
      portOutputs[0].writePtr<cuComplex>());

  const size_t writtenNumBytes = processNumInputElements * sizeof(cuComplex);
  portOutputs[0].end += writtenNumBytes;
}
