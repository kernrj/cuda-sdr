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

class Buffer {
 public:
  Buffer() : mCapacity(0), mOffset(0), mEnd(0) {}
  Buffer(size_t capacity, size_t offset, size_t end)
      : mCapacity(capacity), mOffset(offset), mEnd(end) {}

  /**
   * The start of the buffer - what readPtr() returns when offset() is 0.
   */
  [[nodiscard]] virtual uint8_t* base() = 0;
  [[nodiscard]] virtual const uint8_t* base() const = 0;

  /**
   * Returns a Buffer with its base() starting at offsetInOriginal, and a
   * capacity of endInOriginal - offsetInOriginal.
   *
   * The returned Buffer will have 0 used bytes, and updates to bounds in the
   * returned buffer are not reflected in the object returning the slice.
   */
  [[nodiscard]] virtual std::shared_ptr<Buffer> slice(
      size_t offsetInOriginal,
      size_t endInOriginal) = 0;

  [[nodiscard]] size_t offset() const { return mOffset; }
  [[nodiscard]] size_t endOffset() const { return mEnd; }

  void increaseOffset(size_t increaseBy) {
    if (increaseBy > used()) {
      throw std::runtime_error(
          "The offset increase [" + std::to_string(increaseBy)
          + "] cannot be greater than the number of used bytes ["
          + std::to_string(used()) + "]");
    }

    mOffset += increaseBy;

    if (mOffset == mEnd) {
        mOffset = 0;
        mEnd = 0;
    }
  }

  void increaseEndOffset(size_t increaseBy) {
    if (increaseBy > remaining()) {
      throw std::runtime_error(
          "The end offset increase [" + std::to_string(increaseBy)
          + "] cannot be greater than the number of remaining bytes ["
          + std::to_string(remaining()) + "]");
    }

    mEnd += increaseBy;
  }

  void setUsedRange(size_t offset, size_t endOffset) {
      if (offset == endOffset) {
          mOffset = 0;
          mEnd = 0;
          return;
      }

    if (offset > endOffset) {
      throw std::runtime_error(
          "Offset [" + std::to_string(offset)
          + "] cannot be greater than the end offset ["
          + std::to_string(endOffset) + "]");
    }

    mOffset = offset;
    mEnd = endOffset;
  }

  [[nodiscard]] size_t used() const { return mEnd - mOffset; }
  [[nodiscard]] size_t remaining() const { return mCapacity - mEnd; }
  [[nodiscard]] size_t capacity() const { return mCapacity; }
  [[nodiscard]] bool hasRemaining() const { return remaining() > 0; }

  void reset() {
    mOffset = 0;
    mEnd = 0;
  }

  [[nodiscard]] std::shared_ptr<Buffer> sliceRemaining() {
    return slice(mEnd, mCapacity);
  }

  template <class T = uint8_t>
  [[nodiscard]] const T* readPtr() const {
    return reinterpret_cast<const T*>(base() + offset());
  }

  template <class T = uint8_t>
  [[nodiscard]] T* writePtr() {
    return reinterpret_cast<T*>(
        reinterpret_cast<uint8_t*>(base()) + endOffset());
  }

 private:
  size_t mCapacity;
  size_t mOffset;
  size_t mEnd;
};

class UnownedBuffer : public Buffer {
 public:
  UnownedBuffer(uint8_t* buffer, size_t capacity, size_t offset, size_t end)
      : Buffer(capacity, offset, end), mBuffer(buffer) {}

  [[nodiscard]] uint8_t* base() override { return mBuffer; }
  [[nodiscard]] const uint8_t* base() const override { return mBuffer; }

  std::shared_ptr<Buffer> slice(size_t offsetInOriginal, size_t endInOriginal)
      override {
    const size_t newCapacity = endInOriginal - offsetInOriginal;

    if (newCapacity > capacity()) {
      throw std::invalid_argument(
          "Requested capacity [" + std::to_string(newCapacity)
          + "] exceeds actual capacity [" + std::to_string(capacity()) + "]");
    }

    return std::make_shared<UnownedBuffer>(
        mBuffer + offsetInOriginal,
        newCapacity,
        0,
        0);
  }

 private:
  uint8_t* mBuffer;
};

class OwnedBuffer : public Buffer {
 public:
  OwnedBuffer() : Buffer() {}
  OwnedBuffer(
      const std::shared_ptr<uint8_t>& buffer,
      size_t capacity,
      size_t offset,
      size_t end)
      : Buffer(capacity, offset, end), mBuffer(buffer) {}

  [[nodiscard]] uint8_t* base() override { return mBuffer.get(); }
  [[nodiscard]] const uint8_t* base() const override { return mBuffer.get(); }

  [[nodiscard]] std::shared_ptr<Buffer> slice(
      size_t offsetInOriginal,
      size_t endInOriginal) override {
    const size_t newCapacity = endInOriginal - offsetInOriginal;

    if (newCapacity > capacity()) {
      throw std::invalid_argument(
          "Requested capacity [" + std::to_string(newCapacity)
          + "] exceeds actual capacity [" + std::to_string(capacity()) + "]");
    }

    const auto slicedBuffer =
        std::shared_ptr<uint8_t>(mBuffer, mBuffer.get() + offsetInOriginal);

    return std::make_shared<OwnedBuffer>(slicedBuffer, newCapacity, 0, newCapacity);
  }

 private:
  std::shared_ptr<uint8_t> mBuffer;
};

std::shared_ptr<Buffer> createBufferWithSize(size_t size);
void ensureMinCapacity(std::shared_ptr<Buffer>* buffer, size_t minSize);
void appendToBuffer(
    Buffer* buffer,
    const void* src,
    size_t count);
void moveUsedToStart(Buffer* buffer);

/**
 * Removes [count] bytes from the beginning of [src] and appends to [dst].
 *
 * [src.offset] is increased, and [dst.endOffset] is decreased by [count].
 *
 * If count exceeds the remaining capacity of [dst], or the available byte count in
 * [src], this method will throw an exception.
 */
void moveFromBuffer(
    Buffer* dst,
    Buffer* src,
    size_t count);

#endif  // SDRTEST_SRC_BUFFER_H_
