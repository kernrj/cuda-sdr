//
// Created by Rick Kern on 1/4/23.
//

#ifndef GPUSDR_BUFFERRANGE_H
#define GPUSDR_BUFFERRANGE_H

#include <cstddef>

#include "buffers/IBufferRangeMutableCapacity.h"

class BufferRange : public IBufferRangeMutableCapacity {
 public:
  BufferRange();

  [[nodiscard]] size_t capacity() const override;
  void setCapacity(size_t capacity) override;

  [[nodiscard]] size_t offset() const override;
  [[nodiscard]] size_t endOffset() const override;

  void setUsedRange(size_t offset, size_t endOffset) override;

 private:
  size_t mCapacity;
  size_t mOffset;
  size_t mEnd;
};

#endif  // GPUSDR_BUFFERRANGE_H
