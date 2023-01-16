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

#ifndef GPUSDR_IBUFFER_H
#define GPUSDR_IBUFFER_H

#include <gpusdrpipeline/buffers/IBufferRange.h>

#include <cstdint>

class IBuffer {
 public:
  virtual ~IBuffer() = default;

  /**
   * The start of the buffer - what readPtr() returns when offset() is 0.
   */
  [[nodiscard]] virtual uint8_t* base() = 0;
  [[nodiscard]] virtual const uint8_t* base() const = 0;

  [[nodiscard]] virtual IBufferRange* range() = 0;
  [[nodiscard]] virtual const IBufferRange* range() const = 0;

  template <class T = uint8_t>
  [[nodiscard]] const T* readPtr() const {
    return reinterpret_cast<const T*>(base() + range()->offset());
  }

  template <class T = uint8_t>
  [[nodiscard]] T* writePtr() {
    return reinterpret_cast<T*>(base() + range()->endOffset());
  }
};

#endif  // GPUSDR_IBUFFER_H
