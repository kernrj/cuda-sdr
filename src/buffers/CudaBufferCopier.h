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

#ifndef GPUSDR_CUDABUFFERCOPIER_H
#define GPUSDR_CUDABUFFERCOPIER_H

#include <cuda_runtime.h>

#include <cstdint>

#include "buffers/IBufferCopier.h"
#include "commandqueue/ICudaCommandQueue.h"

class CudaBufferCopier final : public IBufferCopier {
 public:
  CudaBufferCopier(ICudaCommandQueue* commandQueue, cudaMemcpyKind memcpyKind);

  Status copy(void* dst, const void* src, size_t length) const noexcept final;

 private:
  ConstRef<ICudaCommandQueue> mCommandQueue;
  const cudaMemcpyKind mMemcpyKind;

  REF_COUNTED(CudaBufferCopier);
};

#endif  // GPUSDR_CUDABUFFERCOPIER_H
