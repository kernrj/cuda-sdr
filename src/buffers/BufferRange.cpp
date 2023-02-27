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

#include "BufferRange.h"

#include <stdexcept>

#include "GSErrors.h"

BufferRange::BufferRange() noexcept
    : mCapacity(0),
      mOffset(0),
      mEnd(0) {}

size_t BufferRange::capacity() const noexcept { return mCapacity; }
void BufferRange::setCapacity(size_t capacity) noexcept { mCapacity = capacity; }
size_t BufferRange::offset() const noexcept { return mOffset; }
size_t BufferRange::endOffset() const noexcept { return mEnd; }

Status BufferRange::setUsedRange(size_t offset, size_t endOffset) noexcept {
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
      "Error setting used range: Offset [%zu] cannot be greater than the end offset [%zu]",
      offset,
      endOffset);

  mOffset = offset;
  mEnd = endOffset;

  return Status_Success;
}
