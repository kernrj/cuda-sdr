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

#ifndef GPUSDR_FACTORIES_H
#define GPUSDR_FACTORIES_H

#include <gpusdrpipeline/buffers/IBufferFactory.h>
#include <gpusdrpipeline/buffers/IBufferPool.h>
#include <gpusdrpipeline/buffers/IBufferPoolFactory.h>
#include <gpusdrpipeline/buffers/IBufferRangeFactory.h>
#include <gpusdrpipeline/buffers/IBufferSliceFactory.h>
#include <gpusdrpipeline/buffers/IBufferUtil.h>
#include <gpusdrpipeline/buffers/ICudaAllocatorFactory.h>
#include <gpusdrpipeline/buffers/ICudaBufferCopierFactory.h>
#include <gpusdrpipeline/buffers/ICudaMemSetFactory.h>
#include <gpusdrpipeline/buffers/IMemSet.h>
#include <gpusdrpipeline/buffers/IRelocatableCudaBufferFactory.h>
#include <gpusdrpipeline/buffers/IRelocatableResizableBufferFactory.h>
#include <gpusdrpipeline/buffers/IResizableBufferFactory.h>
#include <gpusdrpipeline/commandqueue/ICommandQueueFactory.h>
#include <gpusdrpipeline/commandqueue/ICudaCommandQueueFactory.h>
#include <gpusdrpipeline/driver/IDriver.h>
#include <gpusdrpipeline/driver/IDriverToDiagramFactory.h>
#include <gpusdrpipeline/driver/IFilterDriverFactory.h>
#include <gpusdrpipeline/driver/ISteppingDriverFactory.h>
#include <gpusdrpipeline/filters/FilterFactories.h>

class IFactories : public virtual IRef {
 public:
  [[nodiscard]] virtual IResizableBufferFactory* getResizableBufferFactory() noexcept = 0;
  [[nodiscard]] virtual ICudaAllocatorFactory* getCudaAllocatorFactory() noexcept = 0;
  [[nodiscard]] virtual IBufferSliceFactory* getBufferSliceFactory() = 0;
  [[nodiscard]] virtual IAllocator* getSysMemAllocator() noexcept = 0;
  [[nodiscard]] virtual IBufferCopier* getSysMemCopier() noexcept = 0;
  [[nodiscard]] virtual ICudaBufferCopierFactory* getCudaBufferCopierFactory() noexcept = 0;
  [[nodiscard]] virtual IBufferUtil* getBufferUtil() noexcept = 0;
  [[nodiscard]] virtual ICudaMemcpyFilterFactory* getCudaMemcpyFilterFactory() noexcept = 0;
  [[nodiscard]] virtual IAacFileWriterFactory* getAacFileWriterFactory() noexcept = 0;
  [[nodiscard]] virtual IAddConstFactory* getAddConstFactory() noexcept = 0;
  [[nodiscard]] virtual IAddConstToVectorLengthFactory* getAddConstToVectorLengthFactory() noexcept = 0;
  [[nodiscard]] virtual ICosineSourceFactory* getCosineSourceFactory() noexcept = 0;
  [[nodiscard]] virtual IFileReaderFactory* getFileReaderFactory() noexcept = 0;
  [[nodiscard]] virtual IFirFactory* getFirFactory() noexcept = 0;
  [[nodiscard]] virtual IHackrfSourceFactory* getHackrfSourceFactory() noexcept = 0;
  [[nodiscard]] virtual ICudaFilterFactory* getInt8ToFloatFactory() noexcept = 0;
  [[nodiscard]] virtual ICudaFilterFactory* getMagnitudeFactory() noexcept = 0;
  [[nodiscard]] virtual ICudaFilterFactory* getMultiplyFactory() noexcept = 0;
  [[nodiscard]] virtual IQuadDemodFactory* getQuadDemodFactory() noexcept = 0;
  [[nodiscard]] virtual IMemSet* getSysMemSet() noexcept = 0;
  [[nodiscard]] virtual ICudaMemSetFactory* getCudaMemSetFactory() noexcept = 0;
  [[nodiscard]] virtual ISteppingDriverFactory* getSteppingDriverFactory() noexcept = 0;
  [[nodiscard]] virtual IFilterDriverFactory* getFilterDriverFactory() noexcept = 0;
  [[nodiscard]] virtual IPortRemappingSinkFactory* getPortRemappingSinkFactory() noexcept = 0;
  [[nodiscard]] virtual IPortRemappingSourceFactory* getPortRemappingSourceFactory() noexcept = 0;
  [[nodiscard]] virtual IRfToPcmAudioFactory* getRfToPcmAudioFactory() noexcept = 0;
  [[nodiscard]] virtual IReadByteCountMonitorFactory* getReadByteCountMonitorFactory() noexcept = 0;
  [[nodiscard]] virtual IDriverToDiagramFactory* getDriverToDotFactory() noexcept = 0;
  [[nodiscard]] virtual IBufferRangeFactory* getBufferRangeFactory() noexcept = 0;
  [[nodiscard]] virtual ICommandQueueFactory* getCommandQueueFactory() noexcept = 0;
  [[nodiscard]] virtual ICudaCommandQueueFactory* getCudaCommandQueueFactory() noexcept = 0;

  [[nodiscard]] virtual Result<IBufferFactory> createBufferFactory(IAllocator* allocator) noexcept = 0;

  [[nodiscard]] virtual Result<IRelocatableResizableBufferFactory> createRelocatableResizableBufferFactory(
      IAllocator* allocator,
      const IBufferCopier* bufferCopier) noexcept = 0;

  [[nodiscard]] virtual Result<IBufferPool> createBufferPool(
      size_t maxBufferCount,
      size_t bufferSize,
      IBufferFactory* bufferFactory) noexcept = 0;

  [[nodiscard]] virtual Result<IBufferPoolFactory> createBufferPoolFactory(
      size_t maxBufferCount,
      IBufferFactory* bufferFactory) noexcept = 0;

  [[nodiscard]] virtual Result<IRelocatableResizableBufferFactory> createRelocatableSysMemBufferFactory() noexcept {
    return createRelocatableResizableBufferFactory(getSysMemAllocator(), getSysMemCopier());
  }

  [[nodiscard]] virtual Result<IRelocatableResizableBufferFactory> createRelocatableCudaBufferFactory(
      ICudaCommandQueue* commandQueue,
      size_t cudaAlignment,
      bool useHostMemory) noexcept {
    ConstRef<ICudaBufferCopierFactory> cudaBufferCopierFactory = getCudaBufferCopierFactory();
    Ref<IAllocator> cudaAllocator;
    Ref<IBufferCopier> cudaMemCopier;

    UNWRAP_OR_FWD_RESULT(
        cudaAllocator,
        getCudaAllocatorFactory()->createCudaAllocator(commandQueue, cudaAlignment, useHostMemory));
    UNWRAP_OR_FWD_RESULT(
        cudaMemCopier,
        cudaBufferCopierFactory->createBufferCopier(commandQueue, cudaMemcpyDeviceToDevice));

    return createRelocatableResizableBufferFactory(cudaAllocator.get(), cudaMemCopier.get());
  }

  [[nodiscard]] virtual Result<IBufferFactory> createSysMemBufferFactory() noexcept {
    return createBufferFactory(getSysMemAllocator());
  }

  ABSTRACT_IREF(IFactories);
};

GS_EXPORT [[nodiscard]] Result<IFactories> getFactoriesSingleton() noexcept;

#endif  // GPUSDR_FACTORIES_H
