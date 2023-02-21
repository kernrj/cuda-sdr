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

#include "BufferPool.h"

using namespace std;

class PoolBuffer final : public IBuffer {
 public:
  PoolBuffer(IBuffer* wrappedBuffer, const shared_ptr<queue<ImmutableRef<IBuffer>>>& availableBuffers)
      : mWrappedBuffer(wrappedBuffer),
        mAvailableBuffers(availableBuffers) {}

  [[nodiscard]] uint8_t* base() noexcept final { return mWrappedBuffer->base(); }
  [[nodiscard]] const uint8_t* base() const noexcept final { return mWrappedBuffer->base(); }
  [[nodiscard]] IBufferRange* range() noexcept final { return mWrappedBuffer->range(); }
  [[nodiscard]] const IBufferRange* range() const noexcept final { return mWrappedBuffer->range(); }

 private:
  const RefCt<PoolBuffer> mRefCt {this, onRefCountZero};
  ConstRef<IBuffer> mWrappedBuffer;
  const weak_ptr<queue<ImmutableRef<IBuffer>>> mAvailableBuffers;

 private:
  ~PoolBuffer() final {
    const auto availableBuffersStrong = mAvailableBuffers.lock();
    if (availableBuffersStrong != nullptr) {
      availableBuffersStrong->push(mWrappedBuffer);
    }
  }

  void ref() const noexcept final { mRefCt.ref(); }
  void unref() const noexcept final { mRefCt.unref(); }

  static void onRefCountZero(PoolBuffer* poolBuffer) noexcept {
    poolBuffer->mWrappedBuffer->range()->clearRange();
    delete poolBuffer;  // Destructor returns it to mAvailableBuffers.
  }
};

BufferPool::BufferPool(size_t maxBufferCount, size_t bufferSize, IBufferFactory* bufferFactory)
    : mMaxBufferCount(maxBufferCount),
      mBufferSize(bufferSize),
      mBufferFactory(bufferFactory),
      mAvailableBuffers(make_shared<queue<ImmutableRef<IBuffer>>>()) {}

size_t BufferPool::getBufferSize() const noexcept { return mBufferSize; }

Result<IBuffer> BufferPool::getBuffer() noexcept {
  unique_lock<mutex> lock(mMutex);

  IBuffer* buffer;
  UNWRAP_OR_FWD_RESULT(buffer, tryGetBufferLocked());

  while (buffer == nullptr) {
    mBufferReturnedCv.wait(lock);
    UNWRAP_OR_FWD_RESULT(buffer, tryGetBufferLocked());
  }

  return makeRefResultNonNull(buffer);
}

Result<IBuffer> BufferPool::tryGetBuffer() noexcept {
  lock_guard<mutex> lock(mMutex);
  return tryGetBufferLocked();
}

Result<IBuffer> BufferPool::tryGetBufferLocked() noexcept {
  if (mAvailableBuffers->empty() && mAllBuffers.size() < mMaxBufferCount) {
    IBuffer* buffer;
    UNWRAP_OR_FWD_RESULT(buffer, mBufferFactory->createBuffer(mBufferSize));

    ImmutableRef<IBuffer> bufferRef(buffer);
    mAllBuffers.push_back(bufferRef);
    mAvailableBuffers->push(bufferRef);
  }

  if (mAvailableBuffers->empty()) {
    return {};
  }

  ConstRef<IBuffer> buffer = mAvailableBuffers->front();
  mAvailableBuffers->pop();

  return createWrapperToReturnToPool(buffer);
}

Result<IBuffer> BufferPool::createWrapperToReturnToPool(IBuffer* originalBuffer) {
  return makeRefResultNonNull<IBuffer>(new (nothrow) PoolBuffer(originalBuffer, mAvailableBuffers));
}
