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

#ifndef GPUSDR_CUDAALLOCATORFACTORY_H
#define GPUSDR_CUDAALLOCATORFACTORY_H

#include "buffers/ICudaAllocatorFactory.h"

class CudaAllocatorFactory final : public ICudaAllocatorFactory {
 public:
  Result<IAllocator> createCudaAllocator(
      int32_t cudaDevice,
      cudaStream_t cudaStream,
      size_t alignment,
      bool useHostMemory) noexcept final;

  REF_COUNTED(CudaAllocatorFactory);
};

#endif  // GPUSDR_CUDAALLOCATORFACTORY_H
