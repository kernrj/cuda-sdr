//
// Created by Rick Kern on 1/4/23.
//

#ifndef GPUSDR_BUFFERUTIL_H
#define GPUSDR_BUFFERUTIL_H

#include "buffers/IBufferUtil.h"

class BufferUtil : public IBufferUtil {
 public:
  ~BufferUtil() override = default;

  void appendToBuffer(
      const std::shared_ptr<IBuffer>& buffer,
      const void* src,
      size_t count,
      const std::shared_ptr<IBufferCopier>& bufferCopier) const override;

  void readFromBuffer(
      void* dst,
      const std::shared_ptr<IBuffer>& buffer,
      size_t count,
      const std::shared_ptr<IBufferCopier>& bufferCopier) const override;

  void moveFromBuffer(
      const std::shared_ptr<IBuffer>& dst,
      const std::shared_ptr<IBuffer>& src,
      size_t count,
      const std::shared_ptr<IBufferCopier>& bufferCopier) const override;
};
#endif  // GPUSDR_BUFFERUTIL_H
