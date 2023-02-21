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

#ifndef GPUSDR_BUFFERPOOL_H
#define GPUSDR_BUFFERPOOL_H

#include <cuda_runtime.h>

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <vector>

#include "buffers/IBufferFactory.h"
#include "buffers/IBufferPool.h"

class BufferPool final : public IBufferPool {
 public:
  BufferPool(size_t maxBufferCount, size_t bufferSize, IBufferFactory* bufferFactory);

  [[nodiscard]] size_t getBufferSize() const noexcept final;
  [[nodiscard]] Result<IBuffer> getBuffer() noexcept final;
  [[nodiscard]] Result<IBuffer> tryGetBuffer() noexcept final;

 private:
  [[nodiscard]] Result<IBuffer> createWrapperToReturnToPool(IBuffer* originalBuffer);

 private:
  const size_t mMaxBufferCount;
  const size_t mBufferSize;
  ConstRef<IBufferFactory> mBufferFactory;
  std::vector<ImmutableRef<IBuffer>> mAllBuffers;
  const std::shared_ptr<std::queue<ImmutableRef<IBuffer>>> mAvailableBuffers;
  std::mutex mMutex;
  std::condition_variable mBufferReturnedCv;

 private:
  Result<IBuffer> tryGetBufferLocked() noexcept;

  REF_COUNTED(BufferPool);
};

#endif  // GPUSDR_BUFFERPOOL_H
