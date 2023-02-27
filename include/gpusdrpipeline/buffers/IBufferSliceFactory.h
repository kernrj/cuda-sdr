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

#ifndef GPUSDR_IBUFFERSLICEFACTORY_H
#define GPUSDR_IBUFFERSLICEFACTORY_H

#include <gpusdrpipeline/Result.h>
#include <gpusdrpipeline/buffers/IBuffer.h>

class IBufferSliceFactory : public virtual IRef {
 public:
  /**
   * Creates a buffer slice, and sets the start and end offsets of the buffer slice to the overlap of the start and end
   * offsets in bufferToSlice.
   */
  [[nodiscard]] virtual Result<IBuffer> slice(
      IBuffer* bufferToSlice,
      size_t sliceStartOffset,
      size_t sliceEndOffset) noexcept = 0;

  /**
   * Returns a buffer containing the unused portion of data at the end of another buffer.
   * The start and end offsets of the returned buffer are set to 0 (overlap with the original buffer is 0).
   */
  [[nodiscard]] Result<IBuffer> sliceRemaining(IBuffer* bufferToSlice) {
    Result<IBuffer> result =
        slice(bufferToSlice, bufferToSlice->range()->endOffset(), bufferToSlice->range()->capacity());

    gslogt(
        "Sliced buffer [base=%p, offset=%zu, endOffset=%zu, capacity=%zu] to [base=%p, offset=%zu, endOffset=%zu, "
        "capacity=%zu]\n",
        bufferToSlice->base(),
        bufferToSlice->range()->offset(),
        bufferToSlice->range()->endOffset(),
        bufferToSlice->range()->capacity(),
        result.value->base(),
        result.value->range()->offset(),
        result.value->range()->endOffset(),
        result.value->range()->capacity());

    return result;
  }

  ABSTRACT_IREF(IBufferSliceFactory);
};

#endif  // GPUSDR_IBUFFERSLICEFACTORY_H
