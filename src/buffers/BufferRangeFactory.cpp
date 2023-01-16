//
// Created by Rick Kern on 1/4/23.
//

#include "BufferRangeFactory.h"

#include "BufferRange.h"

std::shared_ptr<IBufferRangeMutableCapacity> BufferRangeFactory::createBufferRange() {
  return std::make_shared<BufferRange>();
}
