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

#ifndef GPUSDR_RELOCATABLERESIZABLEBUFFER_H
#define GPUSDR_RELOCATABLERESIZABLEBUFFER_H

#include "Factories.h"
#include "buffers/IAllocator.h"
#include "buffers/IBuffer.h"
#include "buffers/IBufferCopier.h"
#include "buffers/IBufferRangeFactory.h"
#include "buffers/IRelocatable.h"
#include "buffers/IRelocatableResizableBuffer.h"

class RelocatableResizableBuffer final : public IRelocatableResizableBuffer {
 public:
  static Result<IRelocatableResizableBuffer> create(
      size_t size,
      IAllocator* allocator,
      const IBufferCopier* bufferCopier,
      const IBufferRangeFactory* bufferRangeFactory) noexcept;

  [[nodiscard]] uint8_t* base() noexcept final;
  [[nodiscard]] const uint8_t* base() const noexcept final;
  [[nodiscard]] IBufferRange* range() noexcept final;
  [[nodiscard]] const IBufferRange* range() const noexcept final;

  Status resize(size_t newSize) noexcept final;

  /**
   * Moves the data starting at srcOffset to dstOffset, sets the offset to dstOffset, and used() length to length.
   *
   * The source and destination ranges can overlap, even if the supplied IBufferCopier does not support overlapping
   * ranges.
   */
  Status relocate(size_t dstOffset, size_t srcOffset, size_t length) noexcept final;

 private:
  ConstRef<IAllocator> mAllocator;
  ConstRef<const IBufferCopier> mBufferCopier;
  ConstRef<IBufferRangeMutableCapacity> mRange;
  Ref<IMemory> mData;
  Ref<IMemory> mDataCopy;

 private:
  RelocatableResizableBuffer(
      const ImmutableRef<IAllocator>& allocator,
      const ImmutableRef<const IBufferCopier>& bufferCopier,
      const ImmutableRef<IBufferRangeMutableCapacity>& bufferRange) noexcept;

  REF_COUNTED(RelocatableResizableBuffer);
};

#endif  // GPUSDR_RELOCATABLERESIZABLEBUFFER_H
