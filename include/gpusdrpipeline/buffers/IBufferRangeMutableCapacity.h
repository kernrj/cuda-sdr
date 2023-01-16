//
// Created by Rick Kern on 1/4/23.
//

#ifndef GPUSDR_IBUFFERRANGEMUTABLECAPACITY_H
#define GPUSDR_IBUFFERRANGEMUTABLECAPACITY_H

#include <gpusdrpipeline/buffers/IBufferRange.h>

class IBufferRangeMutableCapacity : public IBufferRange {
 public:
  ~IBufferRangeMutableCapacity() override = default;
  virtual void setCapacity(size_t capacity) = 0;
};

#endif  // GPUSDR_IBUFFERRANGEMUTABLECAPACITY_H
