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

#include "BufferFactory.h"

#include "GSErrors.h"
#include "OwnedBuffer.h"

using namespace std;

BufferFactory::BufferFactory(IAllocator* allocator, IBufferRangeFactory* bufferRangeFactory) noexcept
    : mAllocator(allocator),
      mBufferRangeFactory(bufferRangeFactory) {}

Result<IBuffer> BufferFactory::createBuffer(size_t size) noexcept {
  ImmutableRef<IMemory> data = unwrap(mAllocator->allocate(size));
  if (data == nullptr) {
    return ERR_RESULT(Status_OutOfMemory);
  }

  IBuffer* buffer;
  UNWRAP_OR_FWD_RESULT(buffer, OwnedBuffer::create(0, 0, data, mBufferRangeFactory));

  return makeRefResultNonNull<IBuffer>(buffer);
}
