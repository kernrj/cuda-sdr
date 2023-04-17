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

#ifndef GPUSDR_RELOCATABLECUDABUFFERFACTORY_H
#define GPUSDR_RELOCATABLECUDABUFFERFACTORY_H

#include "buffers/IBufferRangeFactory.h"
#include "buffers/ICudaAllocatorFactory.h"
#include "buffers/ICudaBufferCopierFactory.h"
#include "buffers/IRelocatableCudaBufferFactory.h"

class RelocatableCudaBufferFactory final : public IRelocatableCudaBufferFactory {
 public:
  RelocatableCudaBufferFactory(
      ICudaAllocatorFactory* allocatorFactory,
      ICudaBufferCopierFactory* cudaBufferCopierFactory,
      IBufferRangeFactory* bufferRangeFactory) noexcept;

  Result<IRelocatableResizableBuffer> createCudaBuffer(
      size_t minSize,
      ICudaCommandQueue* commandQueue,
      size_t alignment,
      bool useHostMemory) noexcept final;

 private:
  ConstRef<ICudaAllocatorFactory> mAllocatorFactory;
  ConstRef<ICudaBufferCopierFactory> mCudaBufferCopierFactory;
  ConstRef<IBufferRangeFactory> mBufferRangeFactory;

  REF_COUNTED(RelocatableCudaBufferFactory);
};

#endif  // GPUSDR_RELOCATABLECUDABUFFERFACTORY_H
