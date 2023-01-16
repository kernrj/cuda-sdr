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

BufferPool::BufferPool(size_t maxBufferCount, size_t bufferSize, const shared_ptr<IBufferFactory>& bufferFactory)
    : mMaxBufferCount(maxBufferCount),
      mBufferSize(bufferSize),
      mBufferFactory(bufferFactory),
      mAvailableBuffers(make_shared<queue<shared_ptr<IBuffer>>>()) {}

size_t BufferPool::getBufferSize() const { return mBufferSize; }

shared_ptr<IBuffer> BufferPool::getBuffer() {
  unique_lock<mutex> lock(mMutex);

  optional<shared_ptr<IBuffer>> buffer = tryGetBufferLocked();

  while (!buffer.has_value()) {
    mBufferReturnedCv.wait(lock);
    buffer = tryGetBufferLocked();
  }

  return buffer.value();
}

std::optional<std::shared_ptr<IBuffer>> BufferPool::tryGetBuffer() {
  lock_guard<mutex> lock(mMutex);
  return tryGetBufferLocked();
}

std::optional<std::shared_ptr<IBuffer>> BufferPool::tryGetBufferLocked() {
  if (mAvailableBuffers->empty() && mAllBuffers.size() < mMaxBufferCount) {
    shared_ptr<IBuffer> buffer = mBufferFactory->createBuffer(mBufferSize);
    mAllBuffers.push_back(buffer);
    mAvailableBuffers->push(buffer);
  }

  if (mAvailableBuffers->empty()) {
    return {};
  }

  shared_ptr<IBuffer> buffer = mAvailableBuffers->front();
  mAvailableBuffers->pop();

  return createSpWrapperToReturnToPool(buffer);
}

std::shared_ptr<IBuffer> BufferPool::createSpWrapperToReturnToPool(const shared_ptr<IBuffer>& originalBuffer) {
  weak_ptr<queue<shared_ptr<IBuffer>>> availableBuffersWeak = mAvailableBuffers;
  auto deleter = [availableBuffersWeak, originalBuffer](IBuffer* rawBuffer) {
    auto availableBuffers = availableBuffersWeak.lock();

    if (availableBuffers == nullptr) {
      return;
    }

    originalBuffer->range()->clearRange();
    availableBuffers->push(originalBuffer);
  };

  /*
   * deleter holds the original shared_ptr<Buffer>. The Buffer won't deallocate if this BufferPool is deallocated
   * before the returned shared_ptr.
   */
  return {originalBuffer.get(), deleter};
}
