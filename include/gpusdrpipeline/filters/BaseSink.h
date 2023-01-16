//
// Created by Rick Kern on 1/5/23.
//

#ifndef GPUSDR_BASESINK_H
#define GPUSDR_BASESINK_H

#include <gpusdrpipeline/buffers/IBufferSliceFactory.h>
#include <gpusdrpipeline/buffers/IRelocatableResizableBuffer.h>
#include <gpusdrpipeline/buffers/IRelocatableResizableBufferFactory.h>
#include <gpusdrpipeline/filters/Filter.h>

#include "buffers/IMemSet.h"

/**
 * Provides requestBuffer() and commitBuffer() methods.
 */
class BaseSink : public virtual Sink {
 public:
  struct InputPort {
    std::shared_ptr<IRelocatableResizableBuffer> inputBuffer;
    bool bufferCheckedOut;
  };

 public:
  BaseSink(
      const std::shared_ptr<IRelocatableResizableBufferFactory>& relocatableResizableBufferFactory,
      const std::shared_ptr<IBufferSliceFactory>& slicedBufferFactory,
      size_t inputPortCount,
      const std::shared_ptr<IMemSet>& memSet = nullptr);
  ~BaseSink() override = default;

  [[nodiscard]] std::shared_ptr<IBuffer> requestBuffer(size_t port, size_t numBytes) override;
  void commitBuffer(size_t port, size_t numBytes) override;

 protected:
  [[nodiscard]] std::shared_ptr<IBuffer> getPortInputBuffer(size_t port);
  [[nodiscard]] std::shared_ptr<const IBuffer> getPortInputBuffer(size_t port) const;

  /**
   * Increases the offset of the port's buffer by numBytes, then moves the available used bytes to the start of the
   * buffer.
   */
  void consumeInputBytesAndMoveUsedToStart(size_t port, size_t numBytes);

 private:
  const std::shared_ptr<IRelocatableResizableBufferFactory> mRelocatableResizableBufferFactory;
  const std::shared_ptr<IBufferSliceFactory> mSlicedBufferFactory;
  std::vector<InputPort> mInputPorts;
  const std::shared_ptr<IMemSet> mMemSet;

 private:
  [[nodiscard]] static std::vector<InputPort> createInputPorts(size_t inputPortCount);
};

#endif  // GPUSDR_BASESINK_H
