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

class ResizableBuffer : public IResizableBuffer {
 public:
  ResizableBuffer(
      size_t initialCapacity,
      size_t startOffset,
      size_t endOffset,
      const std::shared_ptr<IAllocator>& allocator,
      const std::shared_ptr<IBufferCopier>& bufferCopier,
      const std::shared_ptr<IBufferRangeFactory>& bufferRangeFactory);
  ~ResizableBuffer() override = default;

  [[nodiscard]] uint8_t* base() override;
  [[nodiscard]] const uint8_t* base() const override;
  [[nodiscard]] IBufferRange* range() override;
  [[nodiscard]] const IBufferRange* range() const override;

  void resize(size_t newSize, size_t* actualSizeOut) override;

 private:
  const std::shared_ptr<IAllocator> mAllocator;
  const std::shared_ptr<IBufferCopier> mBufferCopier;
  std::shared_ptr<uint8_t> mData;
  const std::shared_ptr<IBufferRangeMutableCapacity> mRange;
};

#endif  // GPUSDR_RESIZABLEBUFFER_H
