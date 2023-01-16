//
// Created by Rick Kern on 1/4/23.
//

#ifndef GPUSDR_SYSMEMBUFFERCOPIER_H
#define GPUSDR_SYSMEMBUFFERCOPIER_H

#include "buffers/IBufferCopier.h"

class SysMemBufferCopier : public IBufferCopier {
 public:
  ~SysMemBufferCopier() override = default;

  void copy(void* dst, const void* src, size_t length) override;
};

#endif  // GPUSDR_SYSMEMBUFFERCOPIER_H
