//
// Created by Rick Kern on 1/14/23.
//

#ifndef GPUSDRPIPELINE_SYSMEMSET_H
#define GPUSDRPIPELINE_SYSMEMSET_H

#include <cstring>

#include "buffers/IMemSet.h"

class SysMemSet : public IMemSet {
 public:
  ~SysMemSet() noexcept override = default;

  void memSet(void* data, uint8_t value, size_t byteCount) override { ::memset(data, value, byteCount); }
};

#endif  // GPUSDRPIPELINE_SYSMEMSET_H
