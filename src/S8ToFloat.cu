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

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "CudaBuffers.h"
#include "CudaDevicePushPop.h"
#include "S8ToFloat.h"
#include "cuda_util.h"

using namespace std;

const size_t CudaInt8ToFloat::mAlignment = 32;

__device__ __forceinline__ float d_int8ToFloat(int8_t value) {
  if (value < -127) {
    value = -127;
  }

  return static_cast<float>(value) / 127.0f;
}

__global__ void k_int8ToFloat(const int8_t* input, float* output) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

  output[x] = d_int8ToFloat(input[x]);
}

CudaInt8ToFloat::CudaInt8ToFloat(int32_t cudaDevice, cudaStream_t cudaStream)
    : mCudaDevice(cudaDevice), mCudaStream(cudaStream), mBufferCheckedOut(false) {}

shared_ptr<Buffer> CudaInt8ToFloat::requestBuffer(size_t port, size_t numBytes) {
  if (port > 0) {
    throw runtime_error("Port [" + to_string(port) + "] is out of range");
  }

  if (mBufferCheckedOut) {
    throw runtime_error("Cannot request buffer - it is already checked out");
  }

  CudaDevicePushPop setAndRestore(mCudaDevice);

  ensureMinCapacityAlignedCuda(&mInputBuffer, numBytes, mAlignment * sizeof(uint8_t), mCudaStream);

  mBufferCheckedOut = true;
  return mInputBuffer->sliceRemaining();
}

void CudaInt8ToFloat::commitBuffer(size_t port, size_t numBytes) {
  if (!mBufferCheckedOut) {
    throw runtime_error("Buffer cannot be committed - it was not checked out");
  }

  mInputBuffer->increaseEndOffset(numBytes);
  mBufferCheckedOut = false;
}

size_t CudaInt8ToFloat::getOutputDataSize(size_t port) {
  if (port != 0) {
    throw invalid_argument("Port [" + to_string(port) + "] is out of range");
  }

  return mInputBuffer->used();
}

size_t CudaInt8ToFloat::getOutputSizeAlignment(size_t port) { return mAlignment; }

void CudaInt8ToFloat::readOutput(const vector<shared_ptr<Buffer>>& portOutputs) {
  if (portOutputs.empty()) {
    throw invalid_argument("One output port is required");
  }

  CudaDevicePushPop setAndRestore(mCudaDevice);

  const auto& outputBuffer = portOutputs[0];

  const size_t elementCount = min(mInputBuffer->used(), outputBuffer->remaining() / sizeof(float));
  dim3 blocks = dim3((elementCount + mAlignment - 1) / mAlignment);
  dim3 threads = dim3(mAlignment);

  k_int8ToFloat<<<blocks, threads, 0, mCudaStream>>>(mInputBuffer->readPtr<int8_t>(), outputBuffer->writePtr<float>());

  outputBuffer->increaseEndOffset(elementCount * sizeof(float));
  mInputBuffer->increaseOffset(elementCount * sizeof(int8_t));
  moveUsedToStartCuda(mInputBuffer.get(), mCudaStream);
}
