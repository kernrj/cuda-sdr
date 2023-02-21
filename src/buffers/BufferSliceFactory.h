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

#ifndef GPUSDR_BUFFERSLICEFACTORY_H
#define GPUSDR_BUFFERSLICEFACTORY_H

#include "BufferSlice.h"
#include "buffers/IBufferSliceFactory.h"

class BufferSliceFactory final : public IBufferSliceFactory {
 public:
  explicit BufferSliceFactory(IBufferRangeFactory* bufferRangeFactory);

  Result<IBuffer> slice(IBuffer* bufferToSlice, size_t sliceStartOffset, size_t sliceEndOffset) noexcept final;

 private:
  ConstRef<IBufferRangeFactory> mBufferRangeFactory;

  REF_COUNTED(BufferSliceFactory);
};

#endif  // GPUSDR_BUFFERSLICEFACTORY_H
