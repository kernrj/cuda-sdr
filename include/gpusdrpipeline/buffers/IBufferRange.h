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

#ifndef GPUSDR_IBUFFERRANGE_H
#define GPUSDR_IBUFFERRANGE_H

#include <gpusdrpipeline/GSDefs.h>
#include <gpusdrpipeline/GSErrors.h>
#include <gpusdrpipeline/IRef.h>
#include <gpusdrpipeline/Status.h>

#include <cstddef>
#include <stdexcept>
#include <string>

class IBufferRange : public virtual IRef {
 public:
  /**
   * Returns the number of bytes available from an IBuffer's base().
   */
  [[nodiscard]] virtual size_t capacity() const noexcept = 0;

  /**
   * Returns the index of the first used byte in a buffer.
   */
  [[nodiscard]] virtual size_t offset() const noexcept = 0;

  /**
   * Returns the exclusive upper bound of the used data in a buffer.
   */
  [[nodiscard]] virtual size_t endOffset() const noexcept = 0;

  /**
   * Sets the range of used bytes: offset and endOffset.
   * offset must be <= endOffset.
   *
   * @param offset
   * @param endOffset
   */
  [[nodiscard]] virtual Status setUsedRange(size_t offset, size_t endOffset) noexcept = 0;

  [[nodiscard]] virtual size_t used() const noexcept { return endOffset() - offset(); }
  [[nodiscard]] virtual size_t remaining() const noexcept { return capacity() - endOffset(); }
  [[nodiscard]] virtual bool hasRemaining() const noexcept { return remaining() > 0; }

  void clearRange() noexcept { (void)setUsedRange(0, 0); }

  [[nodiscard]] Status increaseOffset(size_t increaseBy) {
    const size_t newStartOffset = offset() + increaseBy;
    const size_t currentEndOffset = endOffset();
    if (newStartOffset > currentEndOffset) {
      gsloge("New start offset [%zu] exceeds the end offset [%zu]", newStartOffset, currentEndOffset);
      return Status_InvalidArgument;
    }

    return setUsedRange(newStartOffset, endOffset());
  }

  [[nodiscard]] Status increaseEndOffset(size_t increaseBy) {
    const size_t newEndOffset = endOffset() + increaseBy;
    const size_t currentCapacity = capacity();
    if (newEndOffset > capacity()) {
      gsloge("New end offset [%zu] exceeds the capacity [%zu]", newEndOffset, currentCapacity);
      return Status_InvalidArgument;
    }

    return setUsedRange(offset(), newEndOffset);
  }

  ABSTRACT_IREF(IBufferRange);
};

#endif  // GPUSDR_IBUFFERRANGE_H
