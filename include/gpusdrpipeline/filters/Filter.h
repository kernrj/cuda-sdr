/*
 * Copyright 2022-2023 Rick Kern <kernrj@gmail.com>
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

#include <gpusdrpipeline/Result.h>
#include <gpusdrpipeline/Status.h>
#include <gpusdrpipeline/buffers/IBuffer.h>

class Sink;
class Source;
class IDriver;

class Node : public virtual IRef {
 public:
  virtual Sink* asSink() noexcept { return nullptr; }
  virtual Source* asSource() noexcept { return nullptr; }
  virtual IDriver* asDriver() noexcept { return nullptr; }

  ABSTRACT_IREF(Node);
};

class Sink : public virtual Node {
 public:
  /**
   * Returns a Buffer with the requested number of bytes available to write.
   *
   * A call to commitLastRequestedBuffer() must be made before requestBuffer()
   * can be called again.
   *
   * The returned Buffer's mCapacity may be larger than byteCount. Any extra
   * mCapacity can be freely used.
   */
  [[nodiscard]] virtual Result<IBuffer> requestBuffer(size_t port, size_t byteCount) noexcept = 0;

  /**
   * Commits the mInputBuffer, and specifies the number of bytes actually consumed.
   * The byteCount must be <= the mCapacity of the Buffer returned in the most
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
  [[nodiscard]] virtual Status commitBuffer(size_t port, size_t byteCount) noexcept = 0;
  [[nodiscard]] virtual size_t preferredInputBufferSize(size_t port) noexcept = 0;

  [[nodiscard]] Sink* asSink() noexcept override { return this; }

  ABSTRACT_IREF(Sink);
};

class Source : public virtual Node {
 public:
  /**
   * This returns the number of bytes that will be written in the
   * next call to readOutput() for the given port.
   */
  [[nodiscard]] virtual size_t getOutputDataSize(size_t port) noexcept = 0;

  /**
   * To account for vectorized optimizations, the mCapacity of the Buffer passed
   * to the next call to readOutput() for the given port must be at least the
   * next highest multiple of the returned value above getOutputDataSize() for
   * this port.
   *
   * To calculate the minimum mInputBuffer size, use getNextOutputBufferMinSize().
   */
  [[nodiscard]] virtual size_t getOutputSizeAlignment(size_t port) noexcept = 0;

  /**
   * To read all available data, and to account for vectorized operations, the
   * Buffer passed to the next call to readOutput() for the given port must be
   * at least this size. This value is always >= the value returned by
   * getNextOutputDataSize() for the same port.
   *
   * If a Buffer passed to readOutput() is smaller than this, it may read
   * less than the available amount of data.
   */
  [[nodiscard]] virtual size_t getAlignedOutputDataSize(size_t port) noexcept {
    const size_t alignment = getOutputSizeAlignment(port);
    const size_t outputDataSize = getOutputDataSize(port);

    if (outputDataSize > SIZE_MAX - alignment + 1) {
      return outputDataSize / alignment * alignment;
    } else {
      return (outputDataSize + alignment - 1) / alignment * alignment;
    }
  }

  /**
   * Populates the given Buffers. This method may or may not do some processing.
   * For Sinks, processing may also be done in Sink::commitBuffer().
   *
   * However, this method can avoid a memcpy, because an internal mInputBuffer isn't
   * needed to store the processed results.
   */
  [[nodiscard]] virtual Status readOutput(IBuffer** portOutputBuffers, size_t numPorts) noexcept = 0;

  [[nodiscard]] Source* asSource() noexcept override { return this; }

  ABSTRACT_IREF(Source);
};

class Filter : public virtual Sink, public virtual Source {
  ABSTRACT_IREF(Filter);
};

#endif  // SDRTEST_SRC_FILTER_H_
