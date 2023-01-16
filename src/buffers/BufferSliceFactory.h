//
// Created by Rick Kern on 1/4/23.
//

#ifndef GPUSDR_BUFFERSLICEFACTORY_H
#define GPUSDR_BUFFERSLICEFACTORY_H

#include "BufferSlice.h"
#include "buffers/IBufferSliceFactory.h"

class BufferSliceFactory : public IBufferSliceFactory {
 public:
  explicit BufferSliceFactory(const std::shared_ptr<IBufferRangeFactory>& bufferRangeFactory);
  ~BufferSliceFactory() override = default;

  std::shared_ptr<IBuffer> slice(
      const std::shared_ptr<IBuffer>& bufferToSlice,
      size_t sliceStartOffset,
      size_t sliceEndOffset) override;

 private:
  const std::shared_ptr<IBufferRangeFactory> mBufferRangeFactory;
};

#endif  // GPUSDR_BUFFERSLICEFACTORY_H
