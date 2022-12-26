/*
 * Copyright 2022 Rick Kern <kernrj@gmail.com>
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <cstring>

#include "Buffer.h"
#include "cuda_util.h"

using namespace std;

shared_ptr<Buffer> createAlignedBufferCuda(size_t bufferSize, size_t sizeAlignment, cudaStream_t cudaStream) {
  printf("CUDA memory allocation. Size [%zd] alignment [%zd] stream [%p]\n", bufferSize, sizeAlignment, cudaStream);
  static constexpr size_t addressAlignment = 128;

  uint8_t* rawBuffer = nullptr;

  const size_t usableLength = (bufferSize + sizeAlignment - 1) / sizeAlignment * sizeAlignment;
  const size_t allocSize = (addressAlignment - 1) + usableLength;

  SAFE_CUDA(cudaMallocAsync(&rawBuffer, allocSize, cudaStream));

  auto startAddress =
      reinterpret_cast<uint8_t*>(reinterpret_cast<intptr_t>(rawBuffer + addressAlignment - 1) & -addressAlignment);

  return make_shared<OwnedBuffer>(
      std::shared_ptr<uint8_t>(
          std::shared_ptr<uint8_t>(rawBuffer, [cudaStream](uint8_t* buffer) { cudaFreeAsync(buffer, cudaStream); }),
          startAddress),
      usableLength,
      0,
      0);
}

shared_ptr<Buffer> createAlignedBufferCudaHost(size_t bufferSize, size_t sizeAlignment) {
  static constexpr size_t addressAlignment = 128;

  uint8_t* rawBuffer = nullptr;

  const size_t usableLength = (bufferSize + sizeAlignment - 1) / sizeAlignment * sizeAlignment;
  const size_t allocSize = (addressAlignment - 1) + usableLength;

  SAFE_CUDA(cudaHostAlloc(&rawBuffer, allocSize, cudaHostAllocDefault));

  auto startAddress =
      reinterpret_cast<uint8_t*>(reinterpret_cast<intptr_t>(rawBuffer + addressAlignment - 1) & -addressAlignment);

  return make_shared<OwnedBuffer>(
      std::shared_ptr<uint8_t>(
          std::shared_ptr<uint8_t>(rawBuffer, [](uint8_t* buffer) { cudaFreeHost(buffer); }),
          startAddress),
      usableLength,
      0,
      0);
}

void ensureMinCapacityAlignedCuda(
    std::shared_ptr<Buffer>* buffer,
    size_t minSize,
    size_t alignment,
    cudaStream_t cudaStream) {
  if (buffer == nullptr) {
    throw runtime_error("Buffer must be set");
  }

  const size_t minAlignedSize = (minSize + alignment - 1) / alignment * alignment;
  const bool allocateNew = *buffer == nullptr || (*buffer)->capacity() < minAlignedSize;
  const bool copyFromSource = *buffer != nullptr;

  if (!allocateNew) {
    return;
  }

  const shared_ptr<Buffer> newBuffer = createAlignedBufferCuda(minSize, alignment, cudaStream);

  if (copyFromSource) {
    SAFE_CUDA(cudaMemcpyAsync(
        newBuffer->writePtr(),
        (*buffer)->readPtr(),
        (*buffer)->used(),
        cudaMemcpyDeviceToDevice,
        cudaStream));

    newBuffer->setUsedRange(0, (*buffer)->used());
  }

  *buffer = newBuffer;
}

void ensureMinCapacityAlignedCudaHost(std::shared_ptr<Buffer>* buffer, size_t minSize, size_t alignment) {
  if (buffer == nullptr) {
    throw runtime_error("Buffer must be set");
  }

  const size_t minAlignedSize = (minSize + alignment - 1) / alignment * alignment;
  const bool allocateNew = *buffer == nullptr || (*buffer)->capacity() < minAlignedSize;
  const bool copyFromSource = *buffer != nullptr;

  if (!allocateNew) {
    return;
  }

  const shared_ptr<Buffer> newBuffer = createAlignedBufferCudaHost(minSize, alignment);

  if (copyFromSource) {
    memcpy(newBuffer->writePtr(), (*buffer)->readPtr(), (*buffer)->used());

    newBuffer->setUsedRange(0, (*buffer)->used());
  }

  *buffer = newBuffer;
}

void appendToBufferCuda(
    Buffer* buffer,
    const void* src,
    size_t count,
    cudaStream_t cudaStream,
    cudaMemcpyKind memcpyKind) {
  if (count > buffer->remaining()) {
    throw invalid_argument(
        "Cannot copy more bytes [" + to_string(count) + "] than remain [" + to_string(buffer->remaining()));
  }

  SAFE_CUDA(cudaMemcpyAsync(buffer->writePtr(), src, count, memcpyKind, cudaStream));

  buffer->increaseEndOffset(count);
}

void readFromBufferCuda(void* dst, Buffer* buffer, size_t count, cudaStream_t cudaStream) {
  if (count > buffer->used()) {
    throw invalid_argument(
        "Cannot copy more bytes [" + to_string(count) + "] than available [" + to_string(buffer->used()));
  }

  SAFE_CUDA(cudaMemcpyAsync(dst, buffer->readPtr(), count, cudaMemcpyDeviceToDevice, cudaStream));

  buffer->increaseOffset(count);
}

void moveFromBufferCuda(Buffer* dst, Buffer* src, size_t count, cudaStream_t cudaStream, cudaMemcpyKind kind) {
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

  SAFE_CUDA(cudaMemcpyAsync(dstPtr, srcPtr, count, kind, cudaStream));
}

void moveUsedToStartCuda(Buffer* buffer, cudaStream_t cudaStream) {
  if (buffer->offset() == 0) {
    return;
  }

  if (buffer->used() == 0) {
    buffer->setUsedRange(0, 0);
    return;
  }

  SAFE_CUDA(cudaMemcpyAsync(buffer->base(), buffer->readPtr(), buffer->used(), cudaMemcpyDeviceToDevice, cudaStream));

  buffer->setUsedRange(0, buffer->used());
}
