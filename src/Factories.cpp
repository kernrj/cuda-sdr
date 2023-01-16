//
// Created by Rick Kern on 1/3/23.
//

#include "Factories.h"

#include <mutex>
#include <stdexcept>

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
#include "filters/factories/QuadDemodFactory.h"

using namespace std;

class Factories : public IFactories {
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
        mCudaMemSetFactory(new CudaMemSetFactory()) {}

  ~Factories() override = default;

  shared_ptr<IRelocatableResizableBufferFactory> getRelocatableResizableBufferFactory(
      const shared_ptr<IAllocator>& allocator,
      const shared_ptr<IBufferCopier>& bufferCopier) override {
    return make_shared<RelocatableResizableBufferFactory>(allocator, bufferCopier, mBufferRangeFactory);
  }

  shared_ptr<IResizableBufferFactory> getResizableBufferFactory() override { return mResizableBufferFactory; }
  shared_ptr<ICudaAllocatorFactory> getCudaAllocatorFactory() override { return mCudaAllocatorFactory; }
  shared_ptr<IBufferFactory> getBufferFactory(const shared_ptr<IAllocator>& allocator) override {
    return make_shared<BufferFactory>(allocator, mBufferRangeFactory);
  }
  shared_ptr<IBufferSliceFactory> getBufferSliceFactory() override { return mBufferSliceFactory; }
  shared_ptr<IAllocator> getSysMemAllocator() override { return mSysMemAllocator; }
  shared_ptr<IBufferCopier> getSysMemCopier() override { return mSysMemBufferCopier; }
  shared_ptr<ICudaBufferCopierFactory> getCudaBufferCopierFactory() override { return mCudaBufferCopierFactory; }
  shared_ptr<IBufferUtil> getBufferUtil() override { return mBufferUtil; }
  shared_ptr<IBufferPool> createBufferPool(
      size_t maxBufferCount,
      size_t bufferSize,
      const shared_ptr<IBufferFactory>& bufferFactory) override {
    return make_shared<BufferPool>(maxBufferCount, bufferSize, bufferFactory);
  }
  shared_ptr<IBufferPoolFactory> createBufferPoolFactory(
      size_t maxBufferCount,
      const shared_ptr<IBufferFactory>& bufferFactory) override {
    return make_shared<BufferPoolFactory>(maxBufferCount, bufferFactory);
  }

  std::shared_ptr<ICudaMemcpyFilterFactory> getCudaMemcpyFilterFactory() override { return mCudaMemcpyFilterFactory; }
  std::shared_ptr<IAacFileWriterFactory> getAacFileWriterFactory() override { return mAacFileWriterFactory; }
  std::shared_ptr<IAddConstFactory> getAddConstFactory() override { return mAddConstFactory; }

  std::shared_ptr<IAddConstToVectorLengthFactory> getAddConstToVectorLengthFactory() override {
    return mAddConstToVectorLengthFactory;
  }

  std::shared_ptr<ICosineSourceFactory> getCosineSourceFactory() override { return mCosineSourceFactory; }
  std::shared_ptr<IFileReaderFactory> getFileReaderFactory() override { return mFileReaderFactory; }
  std::shared_ptr<IFirFactory> getFirFactory() override { return mFirFactory; }
  std::shared_ptr<IHackrfSourceFactory> getHackrfSourceFactory() override { return mHackrfSourceFactory; }
  std::shared_ptr<ICudaFilterFactory> getInt8ToFloatFactory() override { return mInt8ToFloatFactory; }
  std::shared_ptr<ICudaFilterFactory> getMagnitudeFactory() override { return mMagnitudeFactory; }
  std::shared_ptr<ICudaFilterFactory> getMultiplyFactory() override { return mMultiplyFactory; }
  std::shared_ptr<IQuadDemodFactory> getQuadDemodFactory() override { return mQuadDemodFactory; }

  std::shared_ptr<IMemSet> getSysMemSet() override { return mSysMemSet; }
  std::shared_ptr<ICudaMemSetFactory> getCudaMemSetFactory() override { return mCudaMemSetFactory; }

 private:
  const shared_ptr<IAllocator> mSysMemAllocator;
  const shared_ptr<ICudaAllocatorFactory> mCudaAllocatorFactory;
  const shared_ptr<IBufferRangeFactory> mBufferRangeFactory;
  const shared_ptr<IBufferCopier> mSysMemBufferCopier;
  const shared_ptr<ICudaBufferCopierFactory> mCudaBufferCopierFactory;
  const shared_ptr<IResizableBufferFactory> mResizableBufferFactory;
  const shared_ptr<IRelocatableResizableBufferFactory> mRelocatableResizableBufferFactory;
  const shared_ptr<IBufferSliceFactory> mBufferSliceFactory;
  const shared_ptr<IBufferUtil> mBufferUtil;
  const shared_ptr<ICudaMemcpyFilterFactory> mCudaMemcpyFilterFactory;
  const shared_ptr<IAacFileWriterFactory> mAacFileWriterFactory;
  const shared_ptr<IAddConstFactory> mAddConstFactory;
  const shared_ptr<IAddConstToVectorLengthFactory> mAddConstToVectorLengthFactory;
  const shared_ptr<ICosineSourceFactory> mCosineSourceFactory;
  const shared_ptr<IFileReaderFactory> mFileReaderFactory;
  const shared_ptr<IFirFactory> mFirFactory;
  const shared_ptr<IHackrfSourceFactory> mHackrfSourceFactory;
  const shared_ptr<ICudaFilterFactory> mInt8ToFloatFactory;
  const shared_ptr<ICudaFilterFactory> mMagnitudeFactory;
  const shared_ptr<ICudaFilterFactory> mMultiplyFactory;
  const shared_ptr<IQuadDemodFactory> mQuadDemodFactory;

  const shared_ptr<IMemSet> mSysMemSet;
  const shared_ptr<ICudaMemSetFactory> mCudaMemSetFactory;
};

IFactories* getFactoriesSingleton() {
  static once_flag createFactoriesOnce;
  static IFactories* factories = nullptr;

  try {
    call_once(createFactoriesOnce, []() { factories = new Factories(); });
  } catch (exception& e) {
    fprintf(stderr, "Failed to create factories: %s\n", e.what());
    abort();
  }

  return factories;
}
