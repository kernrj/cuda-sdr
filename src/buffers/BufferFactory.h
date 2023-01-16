//
// Created by Rick Kern on 1/3/23.
//

#ifndef GPUSDR_BUFFERFACTORY_H
#define GPUSDR_BUFFERFACTORY_H

#include "buffers/IAllocator.h"
#include "buffers/IBufferFactory.h"
#include "buffers/IBufferRangeFactory.h"

class BufferFactory : public IBufferFactory {
 public:
  explicit BufferFactory(
      const std::shared_ptr<IAllocator>& allocator,
      const std::shared_ptr<IBufferRangeFactory>& bufferRangeFactory);
  ~BufferFactory() override = default;

  std::shared_ptr<IBuffer> createBuffer(size_t size) override;

 private:
  const std::shared_ptr<IAllocator> mAllocator;
  const std::shared_ptr<IBufferRangeFactory> mBufferRangeFactory;
};

#endif  // GPUSDR_BUFFERFACTORY_H
