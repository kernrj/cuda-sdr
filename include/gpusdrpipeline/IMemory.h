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

#ifndef GPUSDRPIPELINE_IMEMORY_H
#define GPUSDRPIPELINE_IMEMORY_H

#include <gpusdrpipeline/GSDefs.h>

#include "IRef.h"

class IMemory : public virtual IRef {
 public:
  [[nodiscard]] virtual uint8_t* data() noexcept = 0;
  [[nodiscard]] virtual const uint8_t* data() const noexcept = 0;
  [[nodiscard]] virtual size_t capacity() const noexcept = 0;

  template <typename T = uint8_t>
  T* as() noexcept {
    return reinterpret_cast<T*>(data());
  }

  template <typename T = uint8_t>
  const T* as() const noexcept {
    return reinterpret_cast<T*>(data());
  }

  ABSTRACT_IREF(IMemory);
};

#endif  // GPUSDRPIPELINE_IMEMORY_H
