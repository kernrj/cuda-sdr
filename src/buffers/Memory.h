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

#ifndef GPUSDRPIPELINE_MEMORY_H
#define GPUSDRPIPELINE_MEMORY_H

#include "IMemory.h"

class Memory final : public IMemory {
 public:
  using Deleter = void (*)(uint8_t* data, void* context) noexcept;

 public:
  Memory(uint8_t* data, size_t capacity, Deleter deleter, void* deleterContext);

  [[nodiscard]] uint8_t* data() noexcept final;
  [[nodiscard]] const uint8_t* data() const noexcept final;
  [[nodiscard]] size_t capacity() const noexcept final;

 private:
  uint8_t* const mData;
  const size_t mCapacity;
  const Deleter mDeleter;
  void* const mDeleterContext;

 private:
  ~Memory() final;

  REF_COUNTED_NO_DESTRUCTOR(Memory);
};

#endif  // GPUSDRPIPELINE_MEMORY_H
