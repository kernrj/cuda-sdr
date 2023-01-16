//
// Created by Rick Kern on 1/4/23.
//

#include "OwnedBuffer.h"

using namespace std;

OwnedBuffer::OwnedBuffer(
    size_t capacity,
    size_t offset,
    size_t end,
    const std::shared_ptr<uint8_t>& buffer,
    const shared_ptr<IBufferRangeFactory>& bufferRangeFactory)
    : mBuffer(buffer),
      mRange(bufferRangeFactory->createBufferRangeWithCapacity(capacity)) {
  mRange->setUsedRange(offset, end);
}

uint8_t* OwnedBuffer::base() { return mBuffer.get(); }
const uint8_t* OwnedBuffer::base() const { return mBuffer.get(); }
IBufferRange* OwnedBuffer::range() { return mRange.get(); }
const IBufferRange* OwnedBuffer::range() const { return mRange.get(); }
