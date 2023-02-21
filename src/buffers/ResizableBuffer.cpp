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

Result<IResizableBuffer> ResizableBuffer::create(
    size_t initialCapacity,
    size_t startOffset,
    size_t endOffset,
    IAllocator* allocator,
    const IBufferCopier* bufferCopier,
    const IBufferRangeFactory* bufferRangeFactory) noexcept {
  Ref<IMemory> initialBuffer;
  Ref<IBufferRangeMutableCapacity> bufferRange;
  UNWRAP_OR_FWD_RESULT(initialBuffer, allocator->allocate(initialCapacity));
  UNWRAP_OR_FWD_RESULT(bufferRange, bufferRangeFactory->createBufferRange());
  bufferRange->setCapacity(initialCapacity);
  FWD_IN_RESULT_IF_ERR(bufferRange->setUsedRange(startOffset, endOffset));

  DO_OR_RET_ERR_RESULT(return makeRefResultNonNull<IResizableBuffer>(
      new (nothrow) ResizableBuffer(allocator, bufferCopier, bufferRange.get())));
}

ResizableBuffer::ResizableBuffer(
    IAllocator* allocator,
    const IBufferCopier* bufferCopier,
    IBufferRangeMutableCapacity* bufferRange) noexcept
    : mAllocator(allocator),
      mBufferCopier(bufferCopier),
      mRange(bufferRange) {}

uint8_t* ResizableBuffer::base() noexcept { return mData->data(); }
const uint8_t* ResizableBuffer::base() const noexcept { return mData->data(); }
IBufferRange* ResizableBuffer::range() noexcept { return mRange.get(); }
const IBufferRange* ResizableBuffer::range() const noexcept { return mRange.get(); }

Status ResizableBuffer::resize(size_t newSize) noexcept {
  const size_t originalCapacity = range()->capacity();
  if (newSize > originalCapacity) {
    size_t copyNumBytes = originalCapacity;
    Ref<IMemory> newData;
    UNWRAP_OR_FWD_STATUS(newData, mAllocator->allocate(newSize));
    FWD_IF_ERR(mBufferCopier->copy(newData.get(), base(), copyNumBytes));

    mRange->setCapacity(newData->capacity());
    mData = newData;
  }

  return Status_Success;
}
