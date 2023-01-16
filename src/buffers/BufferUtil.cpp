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
#include <unordered_map>

#include "GSErrors.h"

using namespace std;

void BufferUtil::appendToBuffer(
    const std::shared_ptr<IBuffer>& buffer,
    const void* src,
    size_t count,
    const shared_ptr<IBufferCopier>& bufferCopier) const {
  if (count > buffer->range()->remaining()) {
    throw invalid_argument(
        "Cannot copy more bytes [" + to_string(count) + "] than remain [" + to_string(buffer->range()->remaining()));
  }

  bufferCopier->copy(buffer->writePtr(), src, count);

  buffer->range()->increaseEndOffset(count);
}

void BufferUtil::readFromBuffer(
    void* dst,
    const shared_ptr<IBuffer>& buffer,
    size_t count,
    const shared_ptr<IBufferCopier>& bufferCopier) const {
  if (count > buffer->range()->used()) {
    throw invalid_argument(
        SSTREAM("Cannot copy more bytes [" << count << "] than available [" << buffer->range()->used() << "]"));
  }

  bufferCopier->copy(dst, buffer->readPtr(), count);
  buffer->range()->increaseOffset(count);
}

void BufferUtil::moveFromBuffer(
    const shared_ptr<IBuffer>& dst,
    const shared_ptr<IBuffer>& src,
    size_t count,
    const shared_ptr<IBufferCopier>& bufferCopier) const {
  if (count > src->range()->used()) {
    throw invalid_argument(
        "Cannot copy more bytes [" + to_string(count) + "] than available in the source ["
        + to_string(src->range()->used()));
  }

  if (count > dst->range()->remaining()) {
    throw invalid_argument(
        "Cannot copy more bytes [" + to_string(count) + "] than remain in the destination ["
        + to_string(dst->range()->remaining()));
  }

  const uint8_t* srcPtr = src->readPtr();
  uint8_t* dstPtr = dst->writePtr();

  src->range()->increaseOffset(count);
  dst->range()->increaseEndOffset(count);

  bufferCopier->copy(dstPtr, srcPtr, count);
}
