//
// Created by Rick Kern on 1/3/23.
//

#ifndef GPUSDR_FACTORIES_H
#define GPUSDR_FACTORIES_H

#include <gpusdrpipeline/buffers/IBufferFactory.h>
#include <gpusdrpipeline/buffers/IBufferPool.h>
#include <gpusdrpipeline/buffers/IBufferPoolFactory.h>
#include <gpusdrpipeline/buffers/IBufferSliceFactory.h>
#include <gpusdrpipeline/buffers/IBufferUtil.h>
#include <gpusdrpipeline/buffers/ICudaAllocatorFactory.h>
#include <gpusdrpipeline/buffers/ICudaBufferCopierFactory.h>
#include <gpusdrpipeline/buffers/ICudaMemSetFactory.h>
#include <gpusdrpipeline/buffers/IMemSet.h>
#include <gpusdrpipeline/buffers/IRelocatableCudaBufferFactory.h>
#include <gpusdrpipeline/buffers/IRelocatableResizableBufferFactory.h>
#include <gpusdrpipeline/buffers/IResizableBufferFactory.h>
#include <gpusdrpipeline/filters/FilterFactories.h>

class IFactories {
 public:
  virtual ~IFactories() = default;

  virtual std::shared_ptr<IBufferFactory> getBufferFactory(const std::shared_ptr<IAllocator>& allocator) = 0;
  virtual std::shared_ptr<IResizableBufferFactory> getResizableBufferFactory() = 0;
  virtual std::shared_ptr<IRelocatableResizableBufferFactory> getRelocatableResizableBufferFactory(
      const std::shared_ptr<IAllocator>& allocator,
      const std::shared_ptr<IBufferCopier>& bufferCopier) = 0;
  virtual std::shared_ptr<ICudaAllocatorFactory> getCudaAllocatorFactory() = 0;
  virtual std::shared_ptr<IBufferSliceFactory> getBufferSliceFactory() = 0;
  virtual std::shared_ptr<IAllocator> getSysMemAllocator() = 0;
  virtual std::shared_ptr<IBufferCopier> getSysMemCopier() = 0;
  virtual std::shared_ptr<ICudaBufferCopierFactory> getCudaBufferCopierFactory() = 0;
  virtual std::shared_ptr<IBufferUtil> getBufferUtil() = 0;
  virtual std::shared_ptr<IBufferPool> createBufferPool(
      size_t maxBufferCount,
      size_t bufferSize,
      const std::shared_ptr<IBufferFactory>& bufferFactory) = 0;
  virtual std::shared_ptr<IBufferPoolFactory> createBufferPoolFactory(
      size_t maxBufferCount,
      const std::shared_ptr<IBufferFactory>& bufferFactory) = 0;

  virtual std::shared_ptr<ICudaMemcpyFilterFactory> getCudaMemcpyFilterFactory() = 0;
  virtual std::shared_ptr<IAacFileWriterFactory> getAacFileWriterFactory() = 0;
  virtual std::shared_ptr<IAddConstFactory> getAddConstFactory() = 0;
  virtual std::shared_ptr<IAddConstToVectorLengthFactory> getAddConstToVectorLengthFactory() = 0;
  virtual std::shared_ptr<ICosineSourceFactory> getCosineSourceFactory() = 0;
  virtual std::shared_ptr<IFileReaderFactory> getFileReaderFactory() = 0;
  virtual std::shared_ptr<IFirFactory> getFirFactory() = 0;
  virtual std::shared_ptr<IHackrfSourceFactory> getHackrfSourceFactory() = 0;
  virtual std::shared_ptr<ICudaFilterFactory> getInt8ToFloatFactory() = 0;
  virtual std::shared_ptr<ICudaFilterFactory> getMagnitudeFactory() = 0;
  virtual std::shared_ptr<ICudaFilterFactory> getMultiplyFactory() = 0;
  virtual std::shared_ptr<IQuadDemodFactory> getQuadDemodFactory() = 0;
  virtual std::shared_ptr<IMemSet> getSysMemSet() = 0;
  virtual std::shared_ptr<ICudaMemSetFactory> getCudaMemSetFactory() = 0;

  virtual std::shared_ptr<IRelocatableResizableBufferFactory> getRelocatableSysMemBufferFactory() {
    return getRelocatableResizableBufferFactory(getSysMemAllocator(), getSysMemCopier());
  }

  virtual std::shared_ptr<IRelocatableResizableBufferFactory> getRelocatableCudaBufferFactory(
      int32_t cudaDevice,
      cudaStream_t cudaStream,
      size_t cudaAlignment,
      bool useHostMemory) {
    auto cudaAllocator =
        getCudaAllocatorFactory()->createCudaAllocator(cudaDevice, cudaStream, cudaAlignment, useHostMemory);

    auto cudaMemCopier =
        getCudaBufferCopierFactory()->createBufferCopier(cudaDevice, cudaStream, cudaMemcpyDeviceToDevice);

    return getRelocatableResizableBufferFactory(cudaAllocator, cudaMemCopier);
  }

  virtual std::shared_ptr<IBufferFactory> createSysMemBufferFactory() { return getBufferFactory(getSysMemAllocator()); }
};

#ifdef __cplusplus
extern "C"
#endif
    IFactories*
    getFactoriesSingleton();

#endif  // GPUSDR_FACTORIES_H
