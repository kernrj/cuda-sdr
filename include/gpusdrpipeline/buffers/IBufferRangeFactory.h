//
// Created by Rick Kern on 1/4/23.
//

#ifndef GPUSDR_IBUFFERRANGEFACTORY_H
#define GPUSDR_IBUFFERRANGEFACTORY_H

#include <gpusdrpipeline/buffers/IBufferRangeMutableCapacity.h>

#include <memory>

class IBufferRangeFactory {
 public:
  virtual ~IBufferRangeFactory() = default;
  virtual std::shared_ptr<IBufferRangeMutableCapacity> createBufferRange() = 0;

  std::shared_ptr<IBufferRange> createBufferRangeWithCapacity(size_t capacity) {
    auto range = createBufferRange();
    range->setCapacity(capacity);

    return range;
  }
};

#endif  // GPUSDR_IBUFFERRANGEFACTORY_H
