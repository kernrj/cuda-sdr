/*
 * Copyright 2023 Rick Kern <kernrj@gmail.com>
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

#include "CudaMemory.h"

CudaMemory::CudaMemory(
    uint8_t* data,
    size_t capacity,
    int32_t cudaDevice,
    cudaStream_t cudaStream,
    Deleter deleter,
    void* deleterContext) noexcept
    : mData(data),
      mCapacity(capacity),
      mCudaDevice(cudaDevice),
      mCudaStream(cudaStream),
      mDeleter(deleter),
      mDeleterContext(deleterContext) {}

CudaMemory::~CudaMemory() {
  if (mDeleter != nullptr && mData != nullptr) {
    mDeleter(mData, mDeleterContext, mCudaDevice, mCudaStream);
  }
}

uint8_t* CudaMemory::data() noexcept { return mData; }
const uint8_t* CudaMemory::data() const noexcept { return mData; }
size_t CudaMemory::capacity() const noexcept { return mCapacity; }
int32_t CudaMemory::cudaDevice() const noexcept { return mCudaDevice; }
cudaStream_t CudaMemory::cudaStream() const noexcept { return mCudaStream; }
