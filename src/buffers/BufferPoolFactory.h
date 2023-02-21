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

#ifndef GPUSDR_BUFFERPOOLFACTORY_H
#define GPUSDR_BUFFERPOOLFACTORY_H

#include "buffers/IBufferFactory.h"
#include "buffers/IBufferPoolFactory.h"

class BufferPoolFactory final : public IBufferPoolFactory {
 public:
  BufferPoolFactory(size_t maxBufferCount, IBufferFactory* bufferFactory) noexcept;

  [[nodiscard]] Result<IBufferPool> createBufferPool(size_t bufferSize) noexcept final;

 private:
  const size_t mMaxBufferCount;
  ConstRef<IBufferFactory> mBufferFactory;

  REF_COUNTED(BufferPoolFactory);
};

#endif  // GPUSDR_BUFFERPOOLFACTORY_H
