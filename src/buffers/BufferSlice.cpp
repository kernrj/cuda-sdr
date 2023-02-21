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
  SliceRange(size_t sliceStart, size_t sliceEnd, IBuffer* parentBuffer)
      : mSliceStart(sliceStart),
        mSliceEnd(sliceEnd),
        mParentBuffer(parentBuffer),
        mOffset(getStartOffset(parentBuffer, sliceStart, sliceEnd)),
        mEndOffset(getEndOffset(parentBuffer, sliceStart, sliceEnd)) {
    if (mSliceStart > sliceEnd) {
      GS_FAIL("Invalid slice range: start [" << mSliceStart << "] end [" << sliceEnd << "]");
    }

    if (sliceEnd > parentBuffer->range()->capacity()) {
      GS_FAIL(
          "Invalid slice end-offset [" << sliceEnd << "]: Maximum value is [" << parentBuffer->range()->capacity()
                                       << "]");
    }
  }

  [[nodiscard]] size_t capacity() const noexcept final {
    // sliceEnd() and sliceStart() are clamped to the paren't capacity.
    return sliceEnd() - sliceStart();
  }
  [[nodiscard]] size_t offset() const noexcept final { return clampToParentCapacity(mOffset); }
  [[nodiscard]] size_t endOffset() const noexcept final { return clampToParentCapacity(mEndOffset); }

  Status setUsedRange(size_t offset, size_t endOffset) noexcept final {
    const size_t currentCapacity = capacity();

    if (offset > endOffset) {
      gslog(
          GSLOG_ERROR,
          "Error setting used range: Offset [%zu] cannot be greater than the end offset [%zu]",
          offset,
          endOffset);

      return Status_InvalidArgument;
    } else if (endOffset > currentCapacity) {
      gslog(
          GSLOG_ERROR,
          "Errorsetting used range: End offset [%zu] cannot be greater than the capacitiy [%zu]",
          endOffset,
          currentCapacity);

      return Status_InvalidArgument;
    }

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
  [[nodiscard]] size_t clampToParentCapacity(size_t index) const { return min(index, capacity()); }
  [[nodiscard]] size_t sliceStart() const { return min(mSliceStart, mParentBuffer->range()->capacity()); }
  [[nodiscard]] size_t sliceEnd() const { return min(mSliceEnd, mParentBuffer->range()->capacity()); }

  REF_COUNTED(SliceRange);
};

Result<IBuffer> BufferSlice::create(
    IBuffer* slicedBuffer,
    size_t sliceStart,
    size_t sliceEnd,
    IBufferRangeFactory* bufferRangeFactory) noexcept {
  Ref<IBufferRangeMutableCapacity> bufferRange;
  UNWRAP_OR_FWD_RESULT(bufferRange, bufferRangeFactory->createBufferRange());
  bufferRange->setCapacity(sliceEnd - sliceStart);
  FWD_IN_RESULT_IF_ERR(bufferRange->setUsedRange(0, bufferRange->capacity()));

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
