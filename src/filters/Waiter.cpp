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

#include "Waiter.h"

Waiter::Waiter(int32_t cudaDevice, cudaStream_t cudaStream) noexcept
    : mCudaDevice(cudaDevice),
      mCudaStream(cudaStream),
      mCudaEvent(nullptr),
      mNextCudaEvent(nullptr) {}

Waiter::~Waiter() {
  if (mCudaEvent != nullptr) {
    cudaEventDestroy(mCudaEvent);
  }

  if (mNextCudaEvent != nullptr) {
    cudaEventDestroy(mNextCudaEvent);
  }
}

Status Waiter::recordNextAndWaitPrevious() noexcept {
  CUDA_DEV_PUSH_POP_OR_RET_STATUS(mCudaDevice);

  if (mNextCudaEvent == nullptr) {
    SAFE_CUDA_OR_RET_STATUS(cudaEventCreate(&mNextCudaEvent));
  }

  SAFE_CUDA_OR_RET_STATUS(cudaEventRecord(mNextCudaEvent, mCudaStream));

  if (mCudaEvent != nullptr) {
    SAFE_CUDA_OR_RET_STATUS(cudaEventSynchronize(mCudaEvent));
  }

  std::swap(mCudaEvent, mNextCudaEvent);

  return Status_Success;
}
