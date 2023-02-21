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

#ifndef GPUSDR_BUFFERRANGE_H
#define GPUSDR_BUFFERRANGE_H

#include <cstddef>

#include "buffers/IBufferRangeMutableCapacity.h"

class BufferRange final : public IBufferRangeMutableCapacity {
 public:
  BufferRange() noexcept;

  [[nodiscard]] size_t capacity() const noexcept final;
  void setCapacity(size_t capacity) noexcept final;

  [[nodiscard]] size_t offset() const noexcept final;
  [[nodiscard]] size_t endOffset() const noexcept final;

  Status setUsedRange(size_t offset, size_t endOffset) noexcept final;

 private:
  size_t mCapacity;
  size_t mOffset;
  size_t mEnd;

  REF_COUNTED(BufferRange);
};

#endif  // GPUSDR_BUFFERRANGE_H
