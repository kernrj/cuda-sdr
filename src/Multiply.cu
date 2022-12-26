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

#include "CudaBuffers.h"
#include "CudaDevicePushPop.h"
#include "Multiply.h"
#include "cuComplexOperatorOverloads.cuh"
#include "cuda_util.h"

using namespace std;

const size_t MultiplyCcc::mAlignment = 32;

template <class IN1_T, class IN2_T, class OUT_T>
__global__ void k_Multiply(const IN1_T* in1, const IN2_T* in2, OUT_T* out) {
  const size_t x = blockDim.x * blockIdx.x + threadIdx.x;

  out[x] = in1[x] * in2[x];
}

MultiplyCcc::MultiplyCcc(int32_t cudaDevice, cudaStream_t cudaStream)
    : mInputBuffers(2), mCudaDevice(cudaDevice), mCudaStream(cudaStream), mBufferCheckedOut(false) {}

shared_ptr<Buffer> MultiplyCcc::requestBuffer(size_t port, size_t numBytes) {
  if (port >= mInputBuffers.size()) {
    throw runtime_error("Port [" + to_string(port) + "] is out of range");
  }

  if (mBufferCheckedOut) {
    throw runtime_error("Cannot request buffer - it is already checked out");
  }

  CudaDevicePushPop setAndRestore(mCudaDevice);
  ensureMinCapacityAlignedCuda(&mInputBuffers[port], numBytes, mAlignment * sizeof(cuComplex), mCudaStream);

  return mInputBuffers[port]->sliceRemaining();
}

void MultiplyCcc::commitBuffer(size_t port, size_t numBytes) {
  if (port >= mInputBuffers.size()) {
    throw runtime_error("Port [" + to_string(port) + "] is out of range");
  }

  if (!mBufferCheckedOut) {
    throw runtime_error("Buffer cannot be committed - it was not checked out");
  }

  mInputBuffers[port]->increaseEndOffset(numBytes);
  mBufferCheckedOut = false;
}

size_t MultiplyCcc::getOutputDataSize(size_t port) { return getAvailableNumInputElements() * sizeof(cuComplex); }

size_t MultiplyCcc::getAvailableNumInputElements() const {
  const size_t port0NumElements = mInputBuffers[0]->used() / sizeof(cuComplex);
  const size_t port1NumElements = mInputBuffers[1]->used() / sizeof(cuComplex);
  const size_t numInputElements = min(port0NumElements, port1NumElements);

  return numInputElements;
}

size_t MultiplyCcc::getOutputSizeAlignment(size_t port) { return mAlignment * sizeof(cuComplex); }

void MultiplyCcc::readOutput(const vector<shared_ptr<Buffer>>& portOutputs) {
  if (portOutputs.empty()) {
    throw runtime_error("One output port is required");
  }

  CudaDevicePushPop setAndRestore(mCudaDevice);

  const size_t numInputElements = getAvailableNumInputElements();
  const auto& outputBuffer = portOutputs[0];
  const size_t maxNumOutputElements = outputBuffer->remaining() / sizeof(cuComplex);

  const size_t maxUnalignedNumElementsToProcess = min(numInputElements, maxNumOutputElements);

  const size_t numBlocks = maxUnalignedNumElementsToProcess / mAlignment;
  const size_t processNumInputElements = numBlocks * mAlignment;

  const dim3 blocks = dim3(numBlocks);
  const dim3 threads = dim3(mAlignment);

  k_Multiply<cuComplex, cuComplex, cuComplex><<<blocks, threads, 0, mCudaStream>>>(
      mInputBuffers[0]->readPtr<cuComplex>(),
      mInputBuffers[1]->readPtr<cuComplex>(),
      portOutputs[0]->writePtr<cuComplex>());

  const size_t writtenNumBytes = processNumInputElements * sizeof(cuComplex);
  outputBuffer->increaseEndOffset(writtenNumBytes);
  mInputBuffers[0]->increaseOffset(writtenNumBytes);
  mInputBuffers[1]->increaseOffset(writtenNumBytes);
}
