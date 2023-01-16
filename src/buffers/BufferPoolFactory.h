//
// Created by Rick Kern on 1/6/23.
//

#ifndef GPUSDR_BUFFERPOOLFACTORY_H
#define GPUSDR_BUFFERPOOLFACTORY_H

#include "buffers/IBufferFactory.h"
#include "buffers/IBufferPoolFactory.h"

class BufferPoolFactory : public IBufferPoolFactory {
 public:
  explicit BufferPoolFactory(size_t maxBufferCount, const std::shared_ptr<IBufferFactory>& bufferFactory);
  ~BufferPoolFactory() override = default;

  [[nodiscard]] std::shared_ptr<IBufferPool> createBufferPool(size_t bufferSize) override;

 private:
  const size_t mMaxBufferCount;
  const std::shared_ptr<IBufferFactory> mBufferFactory;
};

#endif  // GPUSDR_BUFFERPOOLFACTORY_H
