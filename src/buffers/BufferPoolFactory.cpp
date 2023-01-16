//
// Created by Rick Kern on 1/6/23.
//

#include "BufferPoolFactory.h"

#include "BufferPool.h"

using namespace std;

BufferPoolFactory::BufferPoolFactory(size_t maxBufferCount, const std::shared_ptr<IBufferFactory>& bufferFactory)
    : mMaxBufferCount(maxBufferCount),
      mBufferFactory(bufferFactory) {}

std::shared_ptr<IBufferPool> BufferPoolFactory::createBufferPool(size_t bufferSize) {
  return make_shared<BufferPool>(mMaxBufferCount, bufferSize, mBufferFactory);
}
