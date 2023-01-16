//
// Created by Rick Kern on 1/4/23.
//

#ifndef GPUSDR_IBUFFERUTIL_H
#define GPUSDR_IBUFFERUTIL_H

#include <gpusdrpipeline/buffers/IBuffer.h>
#include <gpusdrpipeline/buffers/IBufferCopier.h>

#include <memory>

class IBufferUtil {
 public:
  virtual ~IBufferUtil() = default;

  virtual void appendToBuffer(
      const std::shared_ptr<IBuffer>& buffer,
      const void* src,
      size_t count,
      const std::shared_ptr<IBufferCopier>& bufferCopier) const = 0;

  virtual void readFromBuffer(
      void* dst,
      const std::shared_ptr<IBuffer>& buffer,
      size_t count,
      const std::shared_ptr<IBufferCopier>& bufferCopier) const = 0;

  virtual void moveFromBuffer(
      const std::shared_ptr<IBuffer>& dst,
      const std::shared_ptr<IBuffer>& src,
      size_t count,
      const std::shared_ptr<IBufferCopier>& bufferCopier) const = 0;
};

#endif  // GPUSDR_IBUFFERUTIL_H
