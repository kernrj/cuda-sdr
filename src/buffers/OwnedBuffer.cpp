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

#include "OwnedBuffer.h"

using namespace std;

Result<IBuffer> OwnedBuffer::create(
    size_t offset,
    size_t end,
    IMemory* memory,
    IBufferRangeFactory* bufferRangeFactory) noexcept {
  Ref<IBufferRange> range;
  UNWRAP_OR_FWD_RESULT(range, bufferRangeFactory->createBufferRangeWithCapacity(memory->capacity()));
  FWD_IN_RESULT_IF_ERR(range->setUsedRange(offset, end));

  return makeRefResultNonNull<IBuffer>(new (nothrow) OwnedBuffer(memory, range.get()));
}

OwnedBuffer::OwnedBuffer(IMemory* memory, IBufferRange* bufferRange) noexcept
    : mData(memory),
      mRange(bufferRange) {}

uint8_t* OwnedBuffer::base() noexcept { return mData->data(); }
const uint8_t* OwnedBuffer::base() const noexcept { return mData->data(); }
IBufferRange* OwnedBuffer::range() noexcept { return mRange; }
const IBufferRange* OwnedBuffer::range() const noexcept { return mRange; }
