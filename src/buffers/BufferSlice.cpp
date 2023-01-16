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

static size_t getStartOffset(const std::shared_ptr<IBuffer>& bufferBeingSliced, size_t sliceStart, size_t sliceEnd) {
  if (bufferBeingSliced->range()->offset() <= sliceStart) {
    return 0;
  } else if (bufferBeingSliced->range()->offset() >= sliceEnd) {
    return sliceEnd - sliceStart;
  } else {
    return bufferBeingSliced->range()->offset() - sliceStart;
  }
}

static size_t getEndOffset(const std::shared_ptr<IBuffer>& bufferBeingSliced, size_t sliceStart, size_t sliceEnd) {
  if (bufferBeingSliced->range()->endOffset() >= sliceEnd) {
    return sliceEnd - sliceStart;
  } else if (bufferBeingSliced->range()->endOffset() <= sliceStart) {
    return 0;
  } else {
    return bufferBeingSliced->range()->endOffset() - sliceStart;
  }
}

class SliceRange : public IBufferRange {
 public:
  SliceRange(size_t sliceStart, size_t sliceEnd, const shared_ptr<IBuffer>& parentBuffer)
      : mSliceStart(sliceStart),
        mSliceEnd(sliceEnd),
        mParentBuffer(parentBuffer),
        mOffset(getStartOffset(parentBuffer, sliceStart, sliceEnd)),
        mEndOffset(getEndOffset(parentBuffer, sliceStart, sliceEnd)) {
    if (mSliceStart > sliceEnd) {
      THROW("Invalid slice range: start [" << mSliceStart << "] end [" << sliceEnd << "]");
    }

    if (sliceEnd > parentBuffer->range()->capacity()) {
      THROW(
          "Invalid slice end-offset [" << sliceEnd << "]: Maximum value is [" << parentBuffer->range()->capacity()
                                       << "]");
    }
  }

  ~SliceRange() override = default;

  [[nodiscard]] size_t capacity() const override {
    // sliceEnd() and sliceStart() are clamped to the paren't capacity.
    return sliceEnd() - sliceStart();
  }
  [[nodiscard]] size_t offset() const override { return clampToParentCapacity(mOffset); }
  [[nodiscard]] size_t endOffset() const override { return clampToParentCapacity(mEndOffset); }

  void setUsedRange(size_t offset, size_t endOffset) override {
    mOffset = offset;
    mEndOffset = endOffset;
  }

 private:
  const size_t mSliceStart;
  const size_t mSliceEnd;
  const shared_ptr<IBuffer> mParentBuffer;

  size_t mOffset;
  size_t mEndOffset;

 private:
  [[nodiscard]] size_t clampToParentCapacity(size_t index) const { return min(index, capacity()); }
  [[nodiscard]] size_t sliceStart() const { return min(mSliceStart, mParentBuffer->range()->capacity()); }
  [[nodiscard]] size_t sliceEnd() const { return min(mSliceEnd, mParentBuffer->range()->capacity()); }
};

BufferSlice::BufferSlice(
    const std::shared_ptr<IBuffer>& slicedBuffer,
    size_t sliceStart,
    size_t sliceEnd,
    const shared_ptr<IBufferRangeFactory>& bufferRangeFactory)
    : mSlicedBuffer(slicedBuffer),
      mRange(new SliceRange(sliceStart, sliceEnd, slicedBuffer)),
      mSliceStart(sliceStart) {
  if (mSlicedBuffer == nullptr) {
    THROW("The buffer being sliced must not be null.");
  }
}

uint8_t* BufferSlice::base() { return mSlicedBuffer->base() + mSliceStart; }
const uint8_t* BufferSlice::base() const { return mSlicedBuffer->base() + mSliceStart; }
IBufferRange* BufferSlice::range() { return mRange.get(); }
const IBufferRange* BufferSlice::range() const { return mRange.get(); }
