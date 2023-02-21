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

#include "RelocatableResizableBufferFactory.h"

#include "RelocatableResizableBuffer.h"

using namespace std;

RelocatableResizableBufferFactory::RelocatableResizableBufferFactory(
    IAllocator* allocator,
    const IBufferCopier* bufferCopier,
    const IBufferRangeFactory* bufferRangeFactory) noexcept
    : mAllocator(allocator),
      mBufferCopier(bufferCopier),
      mBufferRangeFactory(bufferRangeFactory) {}

Result<IRelocatableResizableBuffer> RelocatableResizableBufferFactory::createRelocatableBuffer(
    size_t size) const noexcept {
  return RelocatableResizableBuffer::create(size, mAllocator, mBufferCopier, mBufferRangeFactory);
}
