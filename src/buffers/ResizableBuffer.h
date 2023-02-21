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

#ifndef GPUSDR_RESIZABLEBUFFER_H
#define GPUSDR_RESIZABLEBUFFER_H

#include "buffers/IAllocator.h"
#include "buffers/IBufferCopier.h"
#include "buffers/IBufferRangeFactory.h"
#include "buffers/IResizableBuffer.h"

class ResizableBuffer final : public IResizableBuffer {
 public:
  static Result<IResizableBuffer> create(
      size_t initialCapacity,
      size_t startOffset,
      size_t endOffset,
      IAllocator* allocator,
      const IBufferCopier* bufferCopier,
      const IBufferRangeFactory* bufferRangeFactory) noexcept;

  [[nodiscard]] uint8_t* base() noexcept final;
  [[nodiscard]] const uint8_t* base() const noexcept final;
  [[nodiscard]] IBufferRange* range() noexcept final;
  [[nodiscard]] const IBufferRange* range() const noexcept final;

  Status resize(size_t newSize) noexcept final;

 private:
  ConstRef<IAllocator> mAllocator;
  ConstRef<const IBufferCopier> mBufferCopier;
  Ref<IMemory> mData;
  ConstRef<IBufferRangeMutableCapacity> mRange;

 private:
  ResizableBuffer(
      IAllocator* allocator,
      const IBufferCopier* bufferCopier,
      IBufferRangeMutableCapacity* bufferRange) noexcept;

  REF_COUNTED(ResizableBuffer);
};

#endif  // GPUSDR_RESIZABLEBUFFER_H
