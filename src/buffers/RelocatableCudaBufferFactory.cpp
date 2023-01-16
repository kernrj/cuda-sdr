//
// Created by Rick Kern on 1/4/23.
//

#include "RelocatableCudaBufferFactory.h"

#include "RelocatableResizableBuffer.h"

using namespace std;

RelocatableCudaBufferFactory::RelocatableCudaBufferFactory(
    const shared_ptr<ICudaAllocatorFactory>& allocatorFactory,
    const shared_ptr<ICudaBufferCopierFactory>& cudaBufferCopierFactory,
    const shared_ptr<IBufferRangeFactory>& bufferRangeFactory)
    : mAllocatorFactory(allocatorFactory),
      mCudaBufferCopierFactory(cudaBufferCopierFactory),
      mBufferRangeFactory(bufferRangeFactory) {}

shared_ptr<IRelocatableResizableBuffer> RelocatableCudaBufferFactory::createCudaBuffer(
    size_t minSize,
    int32_t cudaDevice,
    cudaStream_t cudaStream,
    size_t alignment,
    bool useHostMemory) {
  const shared_ptr<IAllocator> cudaAllocator =
      mAllocatorFactory->createCudaAllocator(cudaDevice, cudaStream, alignment, useHostMemory);

  const shared_ptr<IBufferCopier> cudaBufferCopier =
      mCudaBufferCopierFactory->createBufferCopier(cudaDevice, cudaStream, cudaMemcpyDeviceToDevice);

  return make_shared<RelocatableResizableBuffer>(minSize, 0, 0, cudaAllocator, cudaBufferCopier, mBufferRangeFactory);
}
