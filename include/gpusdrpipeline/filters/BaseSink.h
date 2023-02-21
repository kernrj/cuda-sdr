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

#ifndef GPUSDR_BASESINK_H
#define GPUSDR_BASESINK_H

#include <gpusdrpipeline/Factories.h>
#include <gpusdrpipeline/Result.h>
#include <gpusdrpipeline/buffers/IBufferSliceFactory.h>
#include <gpusdrpipeline/buffers/IMemSet.h>
#include <gpusdrpipeline/buffers/IRelocatableResizableBuffer.h>
#include <gpusdrpipeline/buffers/IRelocatableResizableBufferFactory.h>
#include <gpusdrpipeline/filters/Filter.h>

#include <vector>

/**
 * Provides requestBuffer() and commitBuffer() methods.
 */
class BaseSink : public virtual Sink {
 public:
  struct InputPort {
    ConstRef<IRelocatableResizableBuffer> inputBuffer;
    bool bufferCheckedOut;
  };

 public:
  BaseSink() = delete;

  [[nodiscard]] Result<IBuffer> requestBuffer(size_t port, size_t numBytes) noexcept override;
  [[nodiscard]] Status commitBuffer(size_t port, size_t byteCount) noexcept override;

 protected:
  BaseSink(
      IRelocatableResizableBufferFactory* relocatableResizableBufferFactory,
      IBufferSliceFactory* slicedBufferFactory,
      size_t inputPortCount,
      IMemSet* memSet = nullptr);

  ~BaseSink() override = default;

 protected:
  [[nodiscard]] Result<IBuffer> getPortInputBuffer(size_t port) noexcept;
  [[nodiscard]] Result<const IBuffer> getPortInputBuffer(size_t port) const noexcept;

  /**
   * Increases the offset of the port's buffer by numBytes, then moves the available used bytes to the start of the
   * buffer.
   */
  [[nodiscard]] Status consumeInputBytesAndMoveUsedToStart(size_t port, size_t numBytes) noexcept;

 private:
  const size_t mInputPortCount;
  ConstRef<IBufferSliceFactory> mSlicedBufferFactory;
  std::vector<InputPort> mInputPorts;
  ConstRef<IMemSet> mMemSet;
  IRelocatableResizableBufferFactory* const mRelocatableResizableBufferFactory;

 private:
  [[nodiscard]] std::vector<InputPort> createInputPorts();
  [[nodiscard]] Status ensureInputPortsInit() noexcept;
};

#endif  // GPUSDR_BASESINK_H
