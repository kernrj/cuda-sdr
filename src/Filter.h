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

#ifndef SDRTEST_SRC_FILTER_H_
#define SDRTEST_SRC_FILTER_H_

#include <string>

#include "Buffer.h"

class Sink {
 public:
  virtual ~Sink() = default;

  /**
   * Returns a Buffer with the requested number of bytes available to write.
   *
   * A call to commitLastRequestedBuffer() must be made before requestBuffer()
   * can be called again.
   *
   * The returned Buffer's capacity may be larger than byteCount. Any extra
   * capacity can be freely used.
   */
  [[nodiscard]] virtual Buffer requestBuffer(size_t port, size_t byteCount) = 0;

  /**
   * Commits the buffer, and specifies the number of bytes actually consumed.
   * The byteCount must be <= the capacity of the Buffer returned in the most
   * recent call to requestBuffer().
   *
   * Committing causes the Sink to process the data.
   *
   * Calling this method with a byteCount = 0 effectively cancels the last
   * call to requestBuffer().
   *
   * Filter implementations may or may not do their processing in this method,
   * as Source::readOutput() is another option and prevents a memcpy.
   */
  virtual void commitBuffer(size_t port, size_t byteCount) = 0;
};

class Source {
 public:
  virtual ~Source() = default;

  /**
   * This returns the number of bytes that will be written in the
   * next call to readOutput() for the given port.
   */
  [[nodiscard]] virtual size_t getOutputDataSize(size_t port) = 0;

  /**
   * To account for vectorized optimizations, the capacity of the Buffer passed
   * to the next call to readOutput() for the given port must be at least the
   * next highest multiple of the returned value above getOutputDataSize() for
   * this port.
   *
   * To calculate the minimum buffer size, use getNextOutputBufferMinSize().
   */
  [[nodiscard]] virtual size_t getOutputSizeAlignment(size_t port) = 0;

  /**
   * To read all available data, and to account for vectorized operations, the
   * Buffer passed to the next call to readOutput() for the given port must be
   * at least this size. This value is always >= the value returned by
   * getNextOutputDataSize() for the same port.
   *
   * If a Buffer passed to readOutput() is smaller than this, it may read
   * less than the available amount of data.
   */
  [[nodiscard]] size_t getAlignedOutputDataSize(size_t port) {
    const auto alignment = getOutputSizeAlignment(port);
    return (getOutputDataSize(port) + alignment - 1) / alignment * alignment;
  }

  /**
   * Populates the given Buffers. This method may or may not do some processing.
   * For Sinks, processing may also be done in Sink::commitBuffer().
   *
   * However, this method can avoid a memcpy, because an internal buffer isn't
   * needed to store the processed results.
   */
  virtual void readOutput(Buffer* portOutputs, size_t portOutputCount) = 0;
};

class Filter : public Sink, public Source {
 public:
  ~Filter() override = default;
};

#endif  // SDRTEST_SRC_FILTER_H_
