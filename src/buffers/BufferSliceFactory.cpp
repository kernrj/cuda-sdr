//
// Created by Rick Kern on 1/4/23.
//

#include "BufferSliceFactory.h"

using namespace std;

BufferSliceFactory::BufferSliceFactory(const shared_ptr<IBufferRangeFactory>& bufferRangeFactory)
    : mBufferRangeFactory(bufferRangeFactory) {}

shared_ptr<IBuffer> BufferSliceFactory::slice(
    const shared_ptr<IBuffer>& bufferToSlice,
    size_t sliceStartOffset,
    size_t sliceEndOffset) {
  return make_shared<BufferSlice>(bufferToSlice, sliceStartOffset, sliceEndOffset, mBufferRangeFactory);
}
