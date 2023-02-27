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

#include "BufferSlice.h"

#include "BufferRange.h"
#include "GSErrors.h"

using namespace std;

static size_t getStartOffset(IBuffer* bufferBeingSliced, size_t sliceStart, size_t sliceEnd) noexcept {
  if (bufferBeingSliced->range()->offset() <= sliceStart) {
    return 0;
  } else if (bufferBeingSliced->range()->offset() >= sliceEnd) {
    return sliceEnd - sliceStart;
  } else {
    return bufferBeingSliced->range()->offset() - sliceStart;
  }
}

static size_t getEndOffset(IBuffer* bufferBeingSliced, size_t sliceStart, size_t sliceEnd) noexcept {
  if (bufferBeingSliced->range()->endOffset() >= sliceEnd) {
    return sliceEnd - sliceStart;
  } else if (bufferBeingSliced->range()->endOffset() <= sliceStart) {
    return 0;
  } else {
    return bufferBeingSliced->range()->endOffset() - sliceStart;
  }
}

class SliceRange final : public IBufferRange {
 public:
  static Result<IBufferRange> create(size_t sliceStart, size_t sliceEnd, IBuffer* parentBuffer) noexcept {
    GS_REQUIRE_OR_RET_RESULT_FMT(
        sliceStart <= sliceEnd,
        "Invalid slice range: start[%zu] end [%zu]",
        sliceStart,
        sliceEnd);

    GS_REQUIRE_OR_RET_RESULT_FMT(
        sliceEnd <= parentBuffer->range()->capacity(),
        "Invalid slice end-offset [%zu]: maximum value is [%zu]",
        sliceEnd,
        parentBuffer->range()->capacity());

    return makeRefResultNonNull<IBufferRange>(new(nothrow) SliceRange(sliceStart, sliceEnd, parentBuffer));
  }

  [[nodiscard]] size_t capacity() const noexcept final {
    // sliceEnd() and sliceStart() are clamped to the paren't capacity.
    return sliceEnd() - sliceStart();
  }
  [[nodiscard]] size_t offset() const noexcept final { return clampToParentCapacity(mOffset); }
  [[nodiscard]] size_t endOffset() const noexcept final { return clampToParentCapacity(mEndOffset); }

  Status setUsedRange(size_t offset, size_t endOffset) noexcept final {
    const size_t currentCapacity = capacity();

    GS_REQUIRE_OR_RET_FMT(
        offset <= endOffset,
        Status_InvalidArgument,
        "Error setting used range: Offset [%zu] cannot be greater than the end offset [%zu]",
        offset,
        endOffset);

    GS_REQUIRE_OR_RET_FMT(
        endOffset <= currentCapacity,
        Status_InvalidArgument,
        "Error setting used range: End offset [%zu] cannot be greater than the capacity [%zu]",
        endOffset,
        currentCapacity);

    mOffset = offset;
    mEndOffset = endOffset;

    return Status_Success;
  }

 private:
  const size_t mSliceStart;
  const size_t mSliceEnd;
  ConstRef<IBuffer> mParentBuffer;

  size_t mOffset;
  size_t mEndOffset;

 private:
  SliceRange(size_t sliceStart, size_t sliceEnd, IBuffer* parentBuffer) noexcept
      : mSliceStart(sliceStart),
        mSliceEnd(sliceEnd),
        mParentBuffer(parentBuffer),
        mOffset(getStartOffset(parentBuffer, sliceStart, sliceEnd)),
        mEndOffset(getEndOffset(parentBuffer, sliceStart, sliceEnd)) {}

  [[nodiscard]] size_t clampToParentCapacity(size_t index) const { return min(index, capacity()); }
  [[nodiscard]] size_t sliceStart() const { return min(mSliceStart, mParentBuffer->range()->capacity()); }
  [[nodiscard]] size_t sliceEnd() const { return min(mSliceEnd, mParentBuffer->range()->capacity()); }

  REF_COUNTED(SliceRange);
};

static size_t clamp(size_t valueToClamp, size_t minValue, size_t maxValue) noexcept {
  return min(max(minValue, valueToClamp), maxValue);
}

void BufferSlice::getSliceOffsetsFromOriginal(
    size_t originalOffset,
    size_t originalEndOffset,
    size_t sliceStart,
    size_t sliceEnd,
    size_t* newSliceOffsetOut,
    size_t* newSliceEndOffsetOut) noexcept {
  if (sliceStart < originalOffset && sliceEnd < originalOffset) {
    *newSliceOffsetOut = 0;
    *newSliceEndOffsetOut = 0;
    return;
  }

  const size_t clampedOffset = clamp(sliceStart, originalOffset, originalEndOffset);
  const size_t clampedEndOffset = clamp(sliceEnd, originalOffset, originalEndOffset);

  *newSliceOffsetOut = clampedOffset - sliceStart;
  *newSliceEndOffsetOut = clampedEndOffset - sliceStart;

  gslogt(
      "Slice calculations - orig offset [%zu] orig end [%zu] slice start [%zu] slice end [%zu] clamped offset [%zu] "
      "clamped end [%zu] new slice offset [%zu] new slice end [%zu]\n",
      originalOffset,
      originalEndOffset,
      sliceStart,
      sliceEnd,
      clampedOffset,
      clampedEndOffset,
      *newSliceOffsetOut,
      *newSliceEndOffsetOut);
}

Result<IBuffer> BufferSlice::create(
    IBuffer* slicedBuffer,
    size_t sliceStart,
    size_t sliceEnd,
    IBufferRangeFactory* bufferRangeFactory) noexcept {
  Ref<IBufferRangeMutableCapacity> bufferRange;
  UNWRAP_OR_FWD_RESULT(bufferRange, bufferRangeFactory->createBufferRange());
  bufferRange->setCapacity(sliceEnd - sliceStart);

  size_t offsetInSlice;
  size_t endOffsetInSlice;
  getSliceOffsetsFromOriginal(
      slicedBuffer->range()->offset(),
      slicedBuffer->range()->endOffset(),
      sliceStart,
      sliceEnd,
      &offsetInSlice,
      &endOffsetInSlice);
  FWD_IN_RESULT_IF_ERR(bufferRange->setUsedRange(offsetInSlice, endOffsetInSlice));

  return makeRefResultNonNull<IBuffer>(new (nothrow) BufferSlice(slicedBuffer, sliceStart, bufferRange.get()));
}

BufferSlice::BufferSlice(IBuffer* slicedBuffer, size_t sliceStart, IBufferRange* bufferRange) noexcept
    : mSlicedBuffer(slicedBuffer),
      mRange(bufferRange),
      mSliceStart(sliceStart) {
  if (mSlicedBuffer == nullptr) {
    GS_FAIL("The buffer being sliced must not be null.");
  }
}

uint8_t* BufferSlice::base() noexcept { return mSlicedBuffer->base() + mSliceStart; }
const uint8_t* BufferSlice::base() const noexcept { return mSlicedBuffer->base() + mSliceStart; }
IBufferRange* BufferSlice::range() noexcept { return mRange.get(); }
const IBufferRange* BufferSlice::range() const noexcept { return mRange.get(); }
