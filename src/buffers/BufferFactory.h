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

#ifndef GPUSDR_BUFFERFACTORY_H
#define GPUSDR_BUFFERFACTORY_H

#include "buffers/IAllocator.h"
#include "buffers/IBufferFactory.h"
#include "buffers/IBufferRangeFactory.h"

class BufferFactory final : public IBufferFactory {
 public:
  explicit BufferFactory(IAllocator* allocator, IBufferRangeFactory* bufferRangeFactory) noexcept;

  Result<IBuffer> createBuffer(size_t size) noexcept final;

 private:
  ConstRef<IAllocator> mAllocator;
  ConstRef<IBufferRangeFactory> mBufferRangeFactory;

  REF_COUNTED(BufferFactory);
};

#endif  // GPUSDR_BUFFERFACTORY_H
