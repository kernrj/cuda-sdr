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

#ifndef GPUSDR_BUFFERSLICE_H
#define GPUSDR_BUFFERSLICE_H

#include "buffers/IBuffer.h"
#include "buffers/IBufferRangeFactory.h"

/*
 * A buffer which passes operations through to another Buffer, except that the capacity, start offset and end offset can
 * vary.
 */
class BufferSlice : public IBuffer {
 public:
  /**
   * Creates a slice of [slicedBuffer]. If the used range of [slicedBuffer] overlaps the slice, the used range in this
   * object is trimmed to a valid range 0 <= offset or endOffset < sliceEnd - sliceStart
   *
   * @param slicedBuffer The buffer to slice.
   * @param sliceStart The start offset in slicedBuffer that our offset 0 will refer to.
   * @param sliceEnd The end index (exclusive). This value must be >= sliceStart.
   */
  BufferSlice(
      const std::shared_ptr<IBuffer>& slicedBuffer,
      size_t sliceStart,
      size_t sliceEnd,
      const std::shared_ptr<IBufferRangeFactory>& bufferRangeFactory);
  ~BufferSlice() override = default;

  [[nodiscard]] uint8_t* base() override;
  [[nodiscard]] const uint8_t* base() const override;

  [[nodiscard]] IBufferRange* range() override;
  [[nodiscard]] const IBufferRange* range() const override;

 private:
  const std::shared_ptr<IBuffer> mSlicedBuffer;
  const std::shared_ptr<IBufferRange> mRange;
  const size_t mSliceStart;
};

#endif  // GPUSDR_BUFFERSLICE_H
