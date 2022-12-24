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

#include <cuda.h>

#include "AddConst.h"
#include "CudaBuffers.h"
#include "CudaDevicePushPop.h"
#include "cuda_util.h"

using namespace std;

const size_t AddConst::mAlignment = 32;

__global__ void k_AddConst(const float* in, float addConst, float* out) {
  const size_t x = blockDim.x * blockIdx.x + threadIdx.x;

  out[x] = addConst + in[x];
}

AddConst::AddConst(float addConst, int32_t cudaDevice, cudaStream_t cudaStream)
    : mAddConst(addConst), mCudaDevice(cudaDevice), mCudaStream(cudaStream),
      mBufferCheckedOut(false) {}

shared_ptr<Buffer> AddConst::requestBuffer(size_t port, size_t numBytes) {
  if (mBufferCheckedOut) {
    throw runtime_error("Cannot request buffer - it is already checked out");
  }

  ensureMinCapacityAlignedCuda(
      &mInputBuffer,
      mInputBuffer->remaining() + numBytes,
      mAlignment * sizeof(float),
      mCudaStream);

  mBufferCheckedOut = true;
  return mInputBuffer->sliceRemaining();
}

void AddConst::commitBuffer(size_t port, size_t numBytes) {
  if (port >= 1) {
    throw runtime_error("Port [" + to_string(port) + "] is out of range");
  }

  if (!mBufferCheckedOut) {
    throw runtime_error("Buffer cannot be committed - it was not checked out");
  }

  mInputBuffer->increaseEndOffset(numBytes);
  mBufferCheckedOut = false;
}

size_t AddConst::getOutputDataSize(size_t port) {
  return getAvailableNumInputElements() * sizeof(float);
}

size_t AddConst::getAvailableNumInputElements() const {
  return mInputBuffer->used() / sizeof(float);
}

size_t AddConst::getOutputSizeAlignment(size_t port) {
  return mAlignment * sizeof(float);
}

void AddConst::readOutput(const std::vector<std::shared_ptr<Buffer>>& portOutputs) {
  if (portOutputs.empty()) {
    throw runtime_error("One output port is required");
  }

  CudaDevicePushPop setAndRestore(mCudaDevice);

  const size_t numInputElements = getAvailableNumInputElements();
  const auto& outputBuffer = portOutputs[0];
  const size_t maxNumOutputElements = outputBuffer->remaining() / sizeof(float);

  const size_t maxUnalignedNumElementsToProcess =
      min(numInputElements, maxNumOutputElements);

  const size_t numBlocks = maxUnalignedNumElementsToProcess / mAlignment;
  const size_t processNumInputElements = numBlocks * mAlignment;

  const dim3 blocks = dim3(numBlocks);
  const dim3 threads = dim3(mAlignment);

  k_AddConst<<<blocks, threads, 0, mCudaStream>>>(
      mInputBuffer->readPtr<float>(),
      mAddConst,
      portOutputs[0]->writePtr<float>());

  const size_t writtenNumBytes = processNumInputElements * sizeof(float);
  portOutputs[0]->increaseEndOffset(writtenNumBytes);
  mInputBuffer->increaseOffset(writtenNumBytes);
  moveUsedToStartCuda(mInputBuffer.get(), mCudaStream);
}
