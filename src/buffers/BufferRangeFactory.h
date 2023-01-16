//
// Created by Rick Kern on 1/4/23.
//

#ifndef GPUSDR_BUFFERRANGEFACTORY_H
#define GPUSDR_BUFFERRANGEFACTORY_H

#include "buffers/IBufferRangeFactory.h"

class BufferRangeFactory : public IBufferRangeFactory {
 public:
  ~BufferRangeFactory() override = default;
  std::shared_ptr<IBufferRangeMutableCapacity> createBufferRange() override;
};

#endif  // GPUSDR_BUFFERRANGEFACTORY_H
