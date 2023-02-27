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

Result<IRelocatableResizableBuffer> RelocatableResizableBuffer::create(
    size_t size,
    IAllocator* allocator,
    const IBufferCopier* bufferCopier,
    const IBufferRangeFactory* bufferRangeFactory) noexcept {
  NON_NULL_PARAM_OR_RET(allocator);
  NON_NULL_PARAM_OR_RET(bufferCopier);
  NON_NULL_PARAM_OR_RET(bufferRangeFactory);

  Ref<IBufferRangeMutableCapacity> bufferRange;
  UNWRAP_OR_FWD_RESULT(bufferRange, bufferRangeFactory->createBufferRange());

  auto buffer = new (nothrow) RelocatableResizableBuffer(allocator, bufferCopier, bufferRange);
  NON_NULL_OR_RET(buffer);
  FWD_IN_RESULT_IF_ERR(buffer->resize(size));

  return makeRefResultNonNull<IRelocatableResizableBuffer>(buffer);
}

RelocatableResizableBuffer::RelocatableResizableBuffer(
    const ImmutableRef<IAllocator>& allocator,
    const ImmutableRef<const IBufferCopier>& bufferCopier,
    const ImmutableRef<IBufferRangeMutableCapacity>& bufferRange) noexcept
    : mAllocator(allocator),
      mBufferCopier(bufferCopier),
      mRange(bufferRange) {}

uint8_t* RelocatableResizableBuffer::base() noexcept { return mData->data(); }
const uint8_t* RelocatableResizableBuffer::base() const noexcept { return mData->data(); }
IBufferRange* RelocatableResizableBuffer::range() noexcept { return mRange.get(); }
const IBufferRange* RelocatableResizableBuffer::range() const noexcept { return mRange.get(); }

Status RelocatableResizableBuffer::resize(size_t newSize) noexcept {
  const size_t originalCapacity = mRange->capacity();
  if (newSize > originalCapacity) {
    mDataCopy.reset();

    size_t copyNumBytes = originalCapacity;

    Result<IMemory> newDataResult = mAllocator->allocate(newSize);
    FWD_IF_ERR(newDataResult.status);
    ConstRef<IMemory> newData = newDataResult.value;

    if (mData != nullptr) {
      FWD_IF_ERR(mBufferCopier->copy(newData->data(), mData->data(), copyNumBytes));
    }

    UNWRAP_OR_FWD_STATUS(mDataCopy, mAllocator->allocate(newData->capacity()));

    mRange->setCapacity(newData->capacity());
    mData = newData;
  }

  return Status_Success;
}

Status RelocatableResizableBuffer::relocate(size_t dstOffset, size_t srcOffset, size_t length) noexcept {
  const size_t capacity = mRange->capacity();
  GS_REQUIRE_OR_RET_STATUS_FMT(
      dstOffset + length <= capacity,
      "Cannot relocate. Target offset [%zu] + length [%zu] exceeds capacity [%zu]",
      dstOffset,
      length,
      capacity);

  GS_REQUIRE_OR_RET_STATUS_FMT(
      srcOffset + length <= capacity,
      "Cannot relocate. Source offset [%zu] + length [%zu] exceeds capacity [%zu]",
      srcOffset,
      length,
      capacity);

  if (length > 0) {
    FWD_IF_ERR(mBufferCopier->copy(mDataCopy->data() + dstOffset, mData->data() + srcOffset, length));
    std::swap(mData, mDataCopy);
  }

  FWD_IF_ERR(mRange->setUsedRange(dstOffset, length));

  return Status_Success;
}
