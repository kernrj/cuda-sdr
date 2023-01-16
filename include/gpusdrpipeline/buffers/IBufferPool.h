//
// Created by Rick Kern on 1/4/23.
//

#ifndef GPUSDR_IBUFFERPOOL_H
#define GPUSDR_IBUFFERPOOL_H

#include <gpusdrpipeline/buffers/IBuffer.h>

#include <optional>

class IBufferPool {
 public:
  virtual ~IBufferPool() = default;

  [[nodiscard]] virtual size_t getBufferSize() const = 0;
  [[nodiscard]] virtual std::shared_ptr<IBuffer> getBuffer() = 0;
  [[nodiscard]] virtual std::optional<std::shared_ptr<IBuffer>> tryGetBuffer() = 0;
};
#endif  // GPUSDR_IBUFFERPOOL_H
