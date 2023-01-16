//
// Created by Rick Kern on 1/14/23.
//

#ifndef GPUSDRPIPELINE_IMEMSET_H
#define GPUSDRPIPELINE_IMEMSET_H

#include <cstddef>
#include <cstdint>

class IMemSet {
 public:
  virtual ~IMemSet() = default;

  virtual void memSet(void* data, uint8_t value, size_t byteCount) = 0;
};

#endif  // GPUSDRPIPELINE_IMEMSET_H
