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

#include "CudaBufferCopier.h"

#include "Result.h"
#include "util/CudaDevicePushPop.h"

CudaBufferCopier::CudaBufferCopier(int32_t cudaDevice, cudaStream_t cudaStream, cudaMemcpyKind memcpyKind)
    : mCudaDevice(cudaDevice),
      mCudaStream(cudaStream),
      mMemcpyKind(memcpyKind) {}

Status CudaBufferCopier::copy(void* dst, const void* src, size_t length) const noexcept {
  if (length > 0 && (dst == nullptr || src == nullptr)) {
    gslog(
        GSLOG_ERROR,
        "Cannot copy CUDA memory: the source [%p] and destination [%p] must be non-null when length [%zu] > 0",
        src,
        dst,
        length);

    return Status_InvalidArgument;
  }

  CUDA_DEV_PUSH_POP_OR_RET_STATUS(mCudaDevice);
  SAFE_CUDA_OR_RET_STATUS(cudaMemcpyAsync(dst, src, length, mMemcpyKind, mCudaStream));

  return Status_Success;
}
