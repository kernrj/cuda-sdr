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

#include "ResizableBuffer.h"

using namespace std;

ResizableBuffer::ResizableBuffer(
    size_t initialCapacity,
    size_t startOffset,
    size_t endOffset,
    const std::shared_ptr<IAllocator>& allocator,
    const std::shared_ptr<IBufferCopier>& bufferCopier,
    const std::shared_ptr<IBufferRangeFactory>& bufferRangeFactory)
    : mAllocator(allocator),
      mBufferCopier(bufferCopier),
      mRange(bufferRangeFactory->createBufferRange()) {
  size_t actualCapacity = 0;
  mData = mAllocator->allocate(initialCapacity, &actualCapacity);
  mRange->setCapacity(actualCapacity);
  mRange->setUsedRange(startOffset, endOffset);
}

uint8_t* ResizableBuffer::base() { return mData.get(); }
const uint8_t* ResizableBuffer::base() const { return mData.get(); }
IBufferRange* ResizableBuffer::range() { return mRange.get(); }
const IBufferRange* ResizableBuffer::range() const { return mRange.get(); }

void ResizableBuffer::resize(size_t newSize, size_t* actualSizeOut) {
  const size_t originalCapacity = range()->capacity();
  if (newSize > originalCapacity) {
    size_t actualCapacity = 0;
    size_t copyNumBytes = originalCapacity;
    size_t newCapacity = 0;
    shared_ptr<uint8_t> newData = mAllocator->allocate(newSize, &newCapacity);
    mBufferCopier->copy(newData.get(), base(), copyNumBytes);

    mRange->setCapacity(newCapacity);
    mData = newData;
  }
}
