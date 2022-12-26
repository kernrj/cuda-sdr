//
// Created by Rick Kern on 10/29/22.
//

#include "Buffer.h"

#include <cstring>

using namespace std;

shared_ptr<Buffer> createBufferWithSize(size_t size) {
  return make_shared<OwnedBuffer>(shared_ptr<uint8_t>(new uint8_t[size], default_delete<uint8_t[]>()), size, 0, 0);
}

void ensureMinCapacity(shared_ptr<Buffer>* buffer, size_t minSize) {
  if (buffer == nullptr) {
    throw runtime_error("ownedBuffer must be set");
  }

  if (*buffer != nullptr && (*buffer)->capacity() >= minSize) {
    return;
  }

  shared_ptr<Buffer> newBuffer = createBufferWithSize(minSize);

  if (*buffer != nullptr) {
    memmove(newBuffer->writePtr(), (*buffer)->readPtr(), (*buffer)->remaining());
    newBuffer->setUsedRange(0, (*buffer)->used());
  }

  *buffer = newBuffer;
}

void appendToBuffer(Buffer* buffer, const void* src, size_t count) {
  if (count > buffer->remaining()) {
    throw invalid_argument(
        "Cannot copy more bytes [" + to_string(count) + "] than available [" + to_string(buffer->used()));
  }

  memmove(buffer->writePtr(), src, count);

  buffer->increaseEndOffset(count);
}

void moveUsedToStart(Buffer* buffer) {
  if (buffer == nullptr) {
    throw runtime_error("Buffer must be set");
  }

  if (buffer->offset() == 0) {
    return;
  }

  if (buffer->used() == 0) {
    buffer->setUsedRange(0, 0);
    return;
  }

  memmove(buffer->base(), buffer->readPtr(), buffer->used());

  buffer->setUsedRange(0, buffer->used());
}

void moveFromBuffer(Buffer* dst, Buffer* src, size_t count) {
  if (count > src->used()) {
    throw invalid_argument(
        "Cannot copy more bytes [" + to_string(count) + "] than available in the source [" + to_string(src->used()));
  }

  if (count > dst->remaining()) {
    throw invalid_argument(
        "Cannot copy more bytes [" + to_string(count) + "] than remain in the destination ["
        + to_string(dst->remaining()));
  }

  const uint8_t* srcPtr = src->readPtr();
  uint8_t* dstPtr = dst->writePtr();

  src->increaseOffset(count);
  dst->increaseEndOffset(count);

  memcpy(dstPtr, srcPtr, count);
}
