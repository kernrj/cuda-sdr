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

#ifndef GPUSDR_OWNEDBUFFER_H
#define GPUSDR_OWNEDBUFFER_H

#include "buffers/IBuffer.h"
#include "buffers/IBufferRangeFactory.h"

class OwnedBuffer : public IBuffer {
 public:
  OwnedBuffer(
      size_t capacity,
      size_t offset,
      size_t end,
      const std::shared_ptr<uint8_t>& buffer,
      const std::shared_ptr<IBufferRangeFactory>& bufferRangeFactory);

  ~OwnedBuffer() override = default;

  [[nodiscard]] uint8_t* base() override;
  [[nodiscard]] const uint8_t* base() const override;
  [[nodiscard]] IBufferRange* range() override;
  [[nodiscard]] const IBufferRange* range() const override;

 private:
  std::shared_ptr<uint8_t> mBuffer;
  std::shared_ptr<IBufferRange> mRange;
};

#endif  // GPUSDR_OWNEDBUFFER_H
