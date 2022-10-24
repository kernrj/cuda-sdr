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

#include "Buffer.h"
#include "cuda_util.h"

OwnedBuffer createAlignedBuffer(
    size_t bufferSize,
    size_t sizeAlignment,
    cudaStream_t cudaStream) {
  static constexpr size_t addressAlignment = 128;

  uint8_t* rawBuffer = nullptr;
  size_t pitch = 0;

  const size_t usableLength =
      (bufferSize + sizeAlignment - 1) / sizeAlignment * sizeAlignment;
  const size_t allocSize = (addressAlignment - 1) + usableLength;

  SAFE_CUDA(cudaMallocAsync(&rawBuffer, allocSize, cudaStream));

  auto startAddress = reinterpret_cast<uint8_t*>(
      reinterpret_cast<intptr_t>(rawBuffer + addressAlignment - 1)
      & -addressAlignment);

  return {
      std::shared_ptr<uint8_t>(
          std::shared_ptr<uint8_t>(
              rawBuffer,
              [cudaStream](uint8_t* buffer) {
                cudaFreeAsync(buffer, cudaStream);
              }),
          startAddress),
      pitch,
      0,
      0};
}

void ensureMinCapacityAligned(
    OwnedBuffer* buffer,
    size_t minSize,
    size_t alignment,
    cudaStream_t cudaStream) {
  const size_t minAlignedSize =
      (minSize + alignment - 1) / alignment * alignment;
  if (buffer->capacity >= minAlignedSize) {
    return;
  }

  OwnedBuffer newBuffer =
      createAlignedBuffer(buffer->used(), alignment, cudaStream);

  SAFE_CUDA(cudaMemcpyAsync(
      newBuffer.writePtr(),
      buffer->readPtr(),
      buffer->used(),
      cudaMemcpyDeviceToDevice,
      cudaStream));

  newBuffer.end = buffer->used();
  std::swap(newBuffer, *buffer);
}
