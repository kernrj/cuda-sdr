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

#include "RelocatableCudaBufferFactory.h"

#include "RelocatableResizableBuffer.h"

using namespace std;

RelocatableCudaBufferFactory::RelocatableCudaBufferFactory(
    ICudaAllocatorFactory* allocatorFactory,
    ICudaBufferCopierFactory* cudaBufferCopierFactory,
    IBufferRangeFactory* bufferRangeFactory) noexcept
    : mAllocatorFactory(allocatorFactory),
      mCudaBufferCopierFactory(cudaBufferCopierFactory),
      mBufferRangeFactory(bufferRangeFactory) {}

Result<IRelocatableResizableBuffer> RelocatableCudaBufferFactory::createCudaBuffer(
    size_t minSize,
    int32_t cudaDevice,
    cudaStream_t cudaStream,
    size_t alignment,
    bool useHostMemory) noexcept {
  Ref<IAllocator> cudaAllocator;

  UNWRAP_OR_FWD_RESULT(
      cudaAllocator,
      mAllocatorFactory->createCudaAllocator(cudaDevice, cudaStream, alignment, useHostMemory));

  Ref<IBufferCopier> cudaBufferCopier;
  UNWRAP_OR_FWD_RESULT(
      cudaBufferCopier,
      mCudaBufferCopierFactory->createBufferCopier(cudaDevice, cudaStream, cudaMemcpyDeviceToDevice));

  return RelocatableResizableBuffer::create(minSize, cudaAllocator.get(), cudaBufferCopier.get(), mBufferRangeFactory);
}
