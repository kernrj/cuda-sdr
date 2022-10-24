/*
 * Copyright 2022 Rick Kern <kernrj@gmail.com>
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

#ifndef SDRTEST_SRC_BUFFER_H_
#define SDRTEST_SRC_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

struct Buffer {
  Buffer() : buffer(nullptr), capacity(0), offset(0), end(0) {}
  Buffer(uint8_t* buffer, size_t capacity, size_t offset, size_t end)
      : buffer(buffer), capacity(capacity), offset(offset), end(end) {}

  uint8_t* buffer;
  size_t capacity;
  size_t offset;
  size_t end;

  [[nodiscard]] size_t used() const { return end - offset; }
  [[nodiscard]] size_t remaining() const { return capacity - end; }

  template <class T = uint8_t>
  [[nodiscard]] const T* readPtr() const {
    return reinterpret_cast<const T*>(buffer + offset);
  }

  template <class T = uint8_t>
  [[nodiscard]] T* writePtr() {
    return reinterpret_cast<T*>(buffer + end);
  }
};

struct OwnedBuffer {
  OwnedBuffer() : capacity(0), offset(0), end(0) {}
  OwnedBuffer(
      const std::shared_ptr<uint8_t>& buffer,
      size_t capacity,
      size_t offset,
      size_t end)
      : buffer(buffer), capacity(capacity), offset(offset), end(end) {}

  std::shared_ptr<uint8_t> buffer;
  size_t capacity;
  size_t offset;
  size_t end;

  [[nodiscard]] size_t used() const { return end - offset; }
  [[nodiscard]] size_t remaining() const { return capacity - end; }
  [[nodiscard]] bool hasRemaining() const { return remaining() > 0; }
  template <class T = uint8_t>
  [[nodiscard]] const T* readPtr() const {
    return reinterpret_cast<const T*>(buffer.get() + offset);
  }

  template <class T = uint8_t>
  [[nodiscard]] T* writePtr() {
    return reinterpret_cast<T*>(buffer.get() + end);
  }

  [[nodiscard]] OwnedBuffer
  slice(size_t newCapacity, size_t offsetInOriginal, size_t endInOriginal) {
    if (newCapacity > capacity) {
      throw std::invalid_argument(
          "Requested capacity [" + std::to_string(newCapacity)
          + "] exceeds actual capacity [" + std::to_string(capacity) + "]");
    }

    const auto slicedBuffer =
        std::shared_ptr<uint8_t>(buffer, buffer.get() + offsetInOriginal);

    return {slicedBuffer, newCapacity, 0, endInOriginal - offsetInOriginal};
  }

  [[nodiscard]] Buffer sliceRemainingUnowned() {
    return sliceUnowned(end, capacity, remaining());
  }

  [[nodiscard]] Buffer sliceUnowned(
      size_t offsetInOriginal,
      size_t endInOriginal,
      size_t newCapacity) {
    if (newCapacity > this->capacity) {
      throw std::invalid_argument(
          "Requested capacity [" + std::to_string(newCapacity)
          + "] exceeds actual capacity [" + std::to_string(capacity) + "]");
    }

    return {
        buffer.get() + offset,
        newCapacity,
        0,
        endInOriginal - offsetInOriginal};
  }
};

#endif  // SDRTEST_SRC_BUFFER_H_
