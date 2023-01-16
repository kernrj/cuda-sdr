//
// Created by Rick Kern on 1/4/23.
//

#ifndef GPUSDR_RELOCATABLECUDABUFFERFACTORY_H
#define GPUSDR_RELOCATABLECUDABUFFERFACTORY_H

#include "buffers/IBufferRangeFactory.h"
#include "buffers/ICudaAllocatorFactory.h"
#include "buffers/ICudaBufferCopierFactory.h"
#include "buffers/IRelocatableCudaBufferFactory.h"

class RelocatableCudaBufferFactory : public IRelocatableCudaBufferFactory {
 public:
  RelocatableCudaBufferFactory(
      const std::shared_ptr<ICudaAllocatorFactory>& allocatorFactory,
      const std::shared_ptr<ICudaBufferCopierFactory>& cudaBufferCopierFactory,
      const std::shared_ptr<IBufferRangeFactory>& bufferRangeFactory);
  ~RelocatableCudaBufferFactory() override = default;

  std::shared_ptr<IRelocatableResizableBuffer> createCudaBuffer(
      size_t minSize,
      int32_t cudaDevice,
      cudaStream_t cudaStream,
      size_t alignment,
      bool useHostMemory) override;

 private:
  const std::shared_ptr<ICudaAllocatorFactory> mAllocatorFactory;
  const std::shared_ptr<ICudaBufferCopierFactory> mCudaBufferCopierFactory;
  const std::shared_ptr<IBufferRangeFactory> mBufferRangeFactory;
};

#endif  // GPUSDR_RELOCATABLECUDABUFFERFACTORY_H
