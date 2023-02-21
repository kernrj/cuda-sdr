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

#include "Factories.h"

#include <mutex>
#include <stdexcept>

#include "GSLog.h"
#include "buffers/BufferFactory.h"
#include "buffers/BufferPool.h"
#include "buffers/BufferPoolFactory.h"
#include "buffers/BufferRangeFactory.h"
#include "buffers/BufferSliceFactory.h"
#include "buffers/BufferUtil.h"
#include "buffers/CudaAllocatorFactory.h"
#include "buffers/CudaBufferCopierFactory.h"
#include "buffers/CudaMemSetFactory.h"
#include "buffers/RelocatableResizableBufferFactory.h"
#include "buffers/ResizableBufferFactory.h"
#include "buffers/SysMemAllocator.h"
#include "buffers/SysMemBufferCopier.h"
#include "buffers/SysMemSet.h"
#include "driver/DriverToDotFactory.h"
#include "driver/FilterDriverFactory.h"
#include "driver/SteppingDriverFactory.h"
#include "filters/FilterFactories.h"
#include "filters/factories/AacFileWriterFactory.h"
#include "filters/factories/AddConstFactory.h"
#include "filters/factories/AddConstToVectorLengthFactory.h"
#include "filters/factories/CosineSourceFactory.h"
#include "filters/factories/CudaMemcpyFilterFactory.h"
#include "filters/factories/FileReaderFactory.h"
#include "filters/factories/FirFactory.h"
#include "filters/factories/HackrfSourceFactory.h"
#include "filters/factories/Int8ToFloatFactory.h"
#include "filters/factories/MagnitudeFactory.h"
#include "filters/factories/MultiplyFactory.h"
#include "filters/factories/PortRemappingSinkFactory.h"
#include "filters/factories/QuadDemodFactory.h"
#include "filters/factories/ReadByteCountMonitorFactory.h"
#include "filters/factories/RfToPcmAudioFactory.h"

using namespace std;

class Factories final : public IFactories {
 public:
  Factories()
      : mSysMemAllocator(new SysMemAllocator()),
        mCudaAllocatorFactory(new CudaAllocatorFactory()),
        mBufferRangeFactory(new BufferRangeFactory()),
        mSysMemBufferCopier(new SysMemBufferCopier()),
        mCudaBufferCopierFactory(new CudaBufferCopierFactory()),
        mResizableBufferFactory(new ResizableBufferFactory(mSysMemAllocator, mSysMemBufferCopier, mBufferRangeFactory)),
        mRelocatableResizableBufferFactory(
            new RelocatableResizableBufferFactory(mSysMemAllocator, mSysMemBufferCopier, mBufferRangeFactory)),
        mBufferSliceFactory(new BufferSliceFactory(mBufferRangeFactory)),
        mBufferUtil(new BufferUtil()),
        mCudaMemcpyFilterFactory(new CudaMemcpyFilterFactory(this)),
        mAacFileWriterFactory(new AacFileWriterFactory(this)),
        mAddConstFactory(new AddConstFactory(this)),
        mAddConstToVectorLengthFactory(new AddConstToVectorLengthFactory(this)),
        mCosineSourceFactory(new CosineSourceFactory()),
        mFileReaderFactory(new FileReaderFactory()),
        mFirFactory(new FirFactory(this)),
        mHackrfSourceFactory(new HackrfSourceFactory(this)),
        mInt8ToFloatFactory(new Int8ToFloatFactory(this)),
        mMagnitudeFactory(new MagnitudeFactory(this)),
        mMultiplyFactory(new MultiplyFactory(this)),
        mQuadDemodFactory(new QuadDemodFactory(this)),
        mSysMemSet(new SysMemSet()),
        mCudaMemSetFactory(new CudaMemSetFactory()),
        mSteppingDriverFactory(new SteppingDriverFactory()),
        mFilterDriverFactory(new FilterDriverFactory(this)),
        mPortRemappingSinkFactory(new PortRemappingSinkFactory()),
        mRfToPcmAudioFactory(new RfToPcmAudioFactory(this)),
        mReadByteCountMonitorFactory(new ReadByteCountMonitorFactory()),
        mDriverToDotFactory(new DriverToDotFactory()) {}

  IResizableBufferFactory* getResizableBufferFactory() noexcept final { return mResizableBufferFactory; }
  ICudaAllocatorFactory* getCudaAllocatorFactory() noexcept final { return mCudaAllocatorFactory; }
  IBufferSliceFactory* getBufferSliceFactory() noexcept final { return mBufferSliceFactory; }
  IAllocator* getSysMemAllocator() noexcept final { return mSysMemAllocator; }
  IBufferCopier* getSysMemCopier() noexcept final { return mSysMemBufferCopier; }
  ICudaBufferCopierFactory* getCudaBufferCopierFactory() noexcept final { return mCudaBufferCopierFactory; }
  IBufferUtil* getBufferUtil() noexcept final { return mBufferUtil; }
  ICudaMemcpyFilterFactory* getCudaMemcpyFilterFactory() noexcept final { return mCudaMemcpyFilterFactory; }
  IAacFileWriterFactory* getAacFileWriterFactory() noexcept final { return mAacFileWriterFactory; }
  IAddConstFactory* getAddConstFactory() noexcept final { return mAddConstFactory; }

  IAddConstToVectorLengthFactory* getAddConstToVectorLengthFactory() noexcept final {
    return mAddConstToVectorLengthFactory;
  }

  ICosineSourceFactory* getCosineSourceFactory() noexcept final { return mCosineSourceFactory; }
  IFileReaderFactory* getFileReaderFactory() noexcept final { return mFileReaderFactory; }
  IFirFactory* getFirFactory() noexcept final { return mFirFactory; }
  IHackrfSourceFactory* getHackrfSourceFactory() noexcept final { return mHackrfSourceFactory; }
  ICudaFilterFactory* getInt8ToFloatFactory() noexcept final { return mInt8ToFloatFactory; }
  ICudaFilterFactory* getMagnitudeFactory() noexcept final { return mMagnitudeFactory; }
  ICudaFilterFactory* getMultiplyFactory() noexcept final { return mMultiplyFactory; }
  IQuadDemodFactory* getQuadDemodFactory() noexcept final { return mQuadDemodFactory; }
  IMemSet* getSysMemSet() noexcept final { return mSysMemSet; }
  ICudaMemSetFactory* getCudaMemSetFactory() noexcept final { return mCudaMemSetFactory; }
  ISteppingDriverFactory* getSteppingDriverFactory() noexcept final { return mSteppingDriverFactory; }
  IFilterDriverFactory* getFilterDriverFactory() noexcept final { return mFilterDriverFactory; }
  IPortRemappingSinkFactory* getPortRemappingSinkFactory() noexcept final { return mPortRemappingSinkFactory; }
  IRfToPcmAudioFactory* getRfToPcmAudioFactory() noexcept final { return mRfToPcmAudioFactory; }
  IReadByteCountMonitorFactory* getReadByteCountMonitorFactory() noexcept final { return mReadByteCountMonitorFactory; }
  IDriverToDiagramFactory* getDriverToDotFactory() noexcept final { return mDriverToDotFactory; }
  IBufferRangeFactory* getBufferRangeFactory() noexcept final { return mBufferRangeFactory; }

  Result<IRelocatableResizableBufferFactory> createRelocatableResizableBufferFactory(
      IAllocator* allocator,
      const IBufferCopier* bufferCopier) noexcept final {
    return makeRefResultNonNull<IRelocatableResizableBufferFactory>(
        new (nothrow) RelocatableResizableBufferFactory(allocator, bufferCopier, mBufferRangeFactory));
  }
  Result<IBufferFactory> createBufferFactory(IAllocator* allocator) noexcept final {
    return makeRefResultNonNull<IBufferFactory>(new (nothrow) BufferFactory(allocator, mBufferRangeFactory));
  }
  Result<IBufferPool> createBufferPool(size_t maxBufferCount, size_t bufferSize, IBufferFactory* bufferFactory) noexcept
      final {
    return makeRefResultNonNull<IBufferPool>(new (nothrow) BufferPool(maxBufferCount, bufferSize, bufferFactory));
  }
  Result<IBufferPoolFactory> createBufferPoolFactory(size_t maxBufferCount, IBufferFactory* bufferFactory) noexcept
      final {
    return makeRefResultNonNull<IBufferPoolFactory>(new (nothrow) BufferPoolFactory(maxBufferCount, bufferFactory));
  }

 private:
  ConstRef<IAllocator> mSysMemAllocator;
  ConstRef<ICudaAllocatorFactory> mCudaAllocatorFactory;
  ConstRef<IBufferRangeFactory> mBufferRangeFactory;
  ConstRef<IBufferCopier> mSysMemBufferCopier;
  ConstRef<ICudaBufferCopierFactory> mCudaBufferCopierFactory;
  ConstRef<IResizableBufferFactory> mResizableBufferFactory;
  ConstRef<IRelocatableResizableBufferFactory> mRelocatableResizableBufferFactory;
  ConstRef<IBufferSliceFactory> mBufferSliceFactory;
  ConstRef<IBufferUtil> mBufferUtil;
  ConstRef<ICudaMemcpyFilterFactory> mCudaMemcpyFilterFactory;
  ConstRef<IAacFileWriterFactory> mAacFileWriterFactory;
  ConstRef<IAddConstFactory> mAddConstFactory;
  ConstRef<IAddConstToVectorLengthFactory> mAddConstToVectorLengthFactory;
  ConstRef<ICosineSourceFactory> mCosineSourceFactory;
  ConstRef<IFileReaderFactory> mFileReaderFactory;
  ConstRef<IFirFactory> mFirFactory;
  ConstRef<IHackrfSourceFactory> mHackrfSourceFactory;
  ConstRef<ICudaFilterFactory> mInt8ToFloatFactory;
  ConstRef<ICudaFilterFactory> mMagnitudeFactory;
  ConstRef<ICudaFilterFactory> mMultiplyFactory;
  ConstRef<IQuadDemodFactory> mQuadDemodFactory;

  ConstRef<IMemSet> mSysMemSet;
  ConstRef<ICudaMemSetFactory> mCudaMemSetFactory;
  ConstRef<ISteppingDriverFactory> mSteppingDriverFactory;
  ConstRef<IFilterDriverFactory> mFilterDriverFactory;
  ConstRef<IPortRemappingSinkFactory> mPortRemappingSinkFactory;
  ConstRef<IRfToPcmAudioFactory> mRfToPcmAudioFactory;
  ConstRef<IReadByteCountMonitorFactory> mReadByteCountMonitorFactory;
  ConstRef<IDriverToDiagramFactory> mDriverToDotFactory;

  REF_COUNTED(Factories);
};

Result<IFactories> getFactoriesSingleton() noexcept {
  static once_flag createFactoriesOnce;
  static Ref<IFactories> factories;

  try {
    call_once(createFactoriesOnce, []() { factories = new (nothrow) Factories(); });
  }
  IF_CATCH_RETURN_RESULT;

  return makeRefResultNonNull<IFactories>(factories.get());
}
