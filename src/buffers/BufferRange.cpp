//
// Created by Rick Kern on 1/4/23.
//

#include "BufferRange.h"

#include <stdexcept>

#include "GSErrors.h"

BufferRange::BufferRange()
    : mCapacity(0),
      mOffset(0),
      mEnd(0) {}

size_t BufferRange::capacity() const { return mCapacity; }
void BufferRange::setCapacity(size_t capacity) { mCapacity = capacity; }
size_t BufferRange::offset() const { return mOffset; }
size_t BufferRange::endOffset() const { return mEnd; }

void BufferRange::setUsedRange(size_t offset, size_t endOffset) {
  if (offset > endOffset) {
    throw std::runtime_error(
        "Offset [" + std::to_string(offset) + "] cannot be greater than the end offset [" + std::to_string(endOffset)
        + "]");
  }

  mOffset = offset;
  mEnd = endOffset;
}
