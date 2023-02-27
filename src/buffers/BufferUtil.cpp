/*
 * Copyright 2022 Rick Kern <kernrj@gmail.com>
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

#include "BufferUtil.h"

#include <cstring>
#include <string>

#include "Result.h"

using namespace std;

Status BufferUtil::appendToBuffer(IBuffer* buffer, const void* src, size_t count, const IBufferCopier* bufferCopier)
    const noexcept {
  const size_t numRemainingBytes = buffer->range()->remaining();
  if (count > numRemainingBytes) {
    gsloge("Cannot copy [%zu] bytes. Only [%zu] bytes remain.", count, numRemainingBytes);
    return Status_InvalidArgument;
  }

  FWD_IF_ERR(bufferCopier->copy(buffer->writePtr(), src, count));
  FWD_IF_ERR(buffer->range()->increaseEndOffset(count));

  return Status_Success;
}

Status BufferUtil::readFromBuffer(void* dst, IBuffer* buffer, size_t count, const IBufferCopier* bufferCopier)
    const noexcept {
  const size_t currentNumUsedBytes = buffer->range()->used();
  if (count > currentNumUsedBytes) {
    gsloge("Cannot copy [%zu] bytes - only [%zu] are available in the source", count, currentNumUsedBytes);
    return Status_InvalidArgument;
  }

  FWD_IF_ERR(bufferCopier->copy(dst, buffer->readPtr(), count));
  FWD_IF_ERR(buffer->range()->increaseOffset(count));

  return Status_Success;
}

Status BufferUtil::moveFromBuffer(IBuffer* dst, IBuffer* src, size_t count, const IBufferCopier* bufferCopier)
    const noexcept {
  const size_t srcNumUsedBytes = src->range()->used();
  const size_t dstNumRemainingBytes = dst->range()->remaining();

  GS_REQUIRE_OR_RET_FMT(
      count <= srcNumUsedBytes,
      Status_InvalidArgument,
      "Cannot copy [%zu] bytes - only [%zu] are available in the source",
      count,
      srcNumUsedBytes);
  GS_REQUIRE_OR_RET_FMT(
      count <= dstNumRemainingBytes,
      Status_InvalidArgument,
      "Cannot copy [%zu] bytes - only [%zu] are remaining in the destination",
      count,
      dstNumRemainingBytes);

  const uint8_t* srcPtr = src->readPtr();
  uint8_t* dstPtr = dst->writePtr();

  FWD_IF_ERR(src->range()->increaseOffset(count));
  FWD_IF_ERR(dst->range()->increaseEndOffset(count));

  FWD_IF_ERR(bufferCopier->copy(dstPtr, srcPtr, count));

  return Status_Success;
}
