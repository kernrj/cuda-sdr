//
// Created by Rick Kern on 1/3/23.
//

#include "BufferFactory.h"

#include "GSErrors.h"
#include "OwnedBuffer.h"

using namespace std;

BufferFactory::BufferFactory(
    const shared_ptr<IAllocator>& allocator,
    const std::shared_ptr<IBufferRangeFactory>& bufferRangeFactory)
    : mAllocator(allocator),
      mBufferRangeFactory(bufferRangeFactory) {}

shared_ptr<IBuffer> BufferFactory::createBuffer(size_t size) {
  size_t actualSize = 0;
  const shared_ptr<uint8_t> data = mAllocator->allocate(size, &actualSize);

  if (actualSize < size) {
    THROW("Actual size of allocated buffer [" << actualSize << "] is less than the requested size [" << size << "]");
  }

  return make_shared<OwnedBuffer>(actualSize, 0, 0, data, mBufferRangeFactory);
}
