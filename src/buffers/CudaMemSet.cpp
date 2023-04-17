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

#include "CudaMemSet.h"

#include "util/CudaDevicePushPop.h"

CudaMemSet::CudaMemSet(ICudaCommandQueue* commandQueue)
    : mCommandQueue(commandQueue) {}

Status CudaMemSet::memSet(void* data, uint8_t value, size_t byteCount) noexcept {
  CUDA_DEV_PUSH_POP_OR_RET_STATUS(mCommandQueue->cudaDevice());
  SAFE_CUDA_OR_RET_STATUS(cudaMemsetAsync(data, value, byteCount, mCommandQueue->cudaStream()));

  return Status_Success;
}
