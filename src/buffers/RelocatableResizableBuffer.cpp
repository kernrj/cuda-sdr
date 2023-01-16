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

#include "RelocatableResizableBuffer.h"

#include "GSErrors.h"

using namespace std;

RelocatableResizableBuffer::RelocatableResizableBuffer(
    size_t initialCapacity,
    size_t startOffset,
    size_t endOffset,
    const std::shared_ptr<IAllocator>& allocator,
    const std::shared_ptr<IBufferCopier>& bufferCopier,
    const std::shared_ptr<IBufferRangeFactory>& bufferRangeFactory)
    : mAllocator(allocator),
      mBufferCopier(bufferCopier),
      mRange(bufferRangeFactory->createBufferRange()) {
  if (startOffset > initialCapacity) {
    THROW(
        "Start offset [" << startOffset << "] cannot be greater than the initial capacity [" << initialCapacity << "]");
  } else if (endOffset > initialCapacity) {
    THROW("End offset [" << startOffset << "] cannot be greater than the initial capacity [" << initialCapacity << "]");
  }

  size_t actualCapacity = 0;
  mData = mAllocator->allocate(initialCapacity, &actualCapacity);
  mDataCopy = mAllocator->allocate(actualCapacity, nullptr);
  mRange->setCapacity(actualCapacity);
  mRange->setUsedRange(startOffset, endOffset);
}

uint8_t* RelocatableResizableBuffer::base() { return mData.get(); }
const uint8_t* RelocatableResizableBuffer::base() const { return mData.get(); }
IBufferRange* RelocatableResizableBuffer::range() { return mRange.get(); }
const IBufferRange* RelocatableResizableBuffer::range() const { return mRange.get(); }

void RelocatableResizableBuffer::resize(size_t newSize, size_t* actualSizeOut) {
  const size_t originalCapacity = mRange->capacity();
  if (newSize > originalCapacity) {
    mDataCopy.reset();

    size_t actualCapacity = 0;
    size_t copyNumBytes = originalCapacity;
    size_t newCapacity = 0;
    shared_ptr<uint8_t> newData = mAllocator->allocate(newSize, &newCapacity);
    mBufferCopier->copy(newData.get(), base(), copyNumBytes);

    mDataCopy = mAllocator->allocate(newCapacity, nullptr);
    mRange->setCapacity(newCapacity);
    mData = newData;
  }
}

void RelocatableResizableBuffer::relocate(size_t dstOffset, size_t srcOffset, size_t length) {
  const size_t capacity = mRange->capacity();
  if (dstOffset + length > capacity) {
    THROW(
        "Cannot copy to range. Destination [" << dstOffset << "] length [" << length << "] - exceeds capacity ["
                                              << capacity << "]");
  } else if (srcOffset + length > capacity) {
    THROW(
        "Cannot copy from range. Source [" << srcOffset << "] length [" << length << "] - exceeds capacity ["
                                           << capacity << "]");
  }

  if (length > 0) {
    mBufferCopier->copy(mDataCopy.get() + dstOffset, mData.get() + srcOffset, length);
    std::swap(mData, mDataCopy);
  }

  mRange->setUsedRange(dstOffset, length);
}
