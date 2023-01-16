//
// Created by Rick Kern on 1/6/23.
//

#ifndef GPUSDR_IBUFFERPOOLFACTORY_H
#define GPUSDR_IBUFFERPOOLFACTORY_H

#include <gpusdrpipeline/buffers/IBufferPool.h>

#include <memory>

class IBufferPoolFactory {
 public:
  virtual ~IBufferPoolFactory() = default;

  [[nodiscard]] virtual std::shared_ptr<IBufferPool> createBufferPool(size_t bufferSize) = 0;
};
#endif  // GPUSDR_IBUFFERPOOLFACTORY_H
