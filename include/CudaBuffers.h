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

#ifndef SDRTEST_SRC_CUDABUFFERS_H_
#define SDRTEST_SRC_CUDABUFFERS_H_

#include <cuda_runtime.h>

#include "Buffer.h"

/**
 * Creates an aligned mInputBuffer, and causes a GPU sync when allocating, and
 * another when the returned mInputBuffer is freed.
 *
 * @param elementCount The number of elements in the returned mInputBuffer.
 */
std::shared_ptr<Buffer> createAlignedBufferCuda(
    size_t bufferSize,
    size_t sizeAlignment,
    cudaStream_t cudaStream);

/**
 * Creates an aligned mInputBuffer, and causes a GPU sync when allocating, and
 * another when the returned mInputBuffer is freed.
 *
 * @param elementCount The number of elements in the returned mInputBuffer.
 */
void ensureMinCapacityAlignedCuda(
    std::shared_ptr<Buffer>* buffer,
    size_t minSize,
    size_t alignment,
    cudaStream_t cudaStream);

void appendToBufferCuda(
    Buffer* buffer,
    const void* src,
    size_t count,
    cudaStream_t cudaStream,
    cudaMemcpyKind memcpyKind);

/**
 * Removes [count] bytes from the beginning of [src] and copies into dst.
 *
 * [src.offset] is increased by [count].
 *
 * If count exceeds the remaining capacity of [dst], or the available byte count in
 * [src], this method will throw an exception.
 */
void readFromBufferCuda(
    void* dst,
    Buffer* buffer,
    size_t count,
    cudaStream_t cudaStream);

/**
 * Removes [count] bytes from the beginning of [src] and appends to [dst].
 *
 * [src.offset] is increased, and [dst.endOffset] is decreased by [count].
 *
 * If count exceeds the remaining capacity of [dst], or the available byte count in
 * [src], this method will throw an exception.
 */
void moveFromBufferCuda(
    Buffer* dst,
    Buffer* src,
    size_t count,
    cudaStream_t cudaStream,
    cudaMemcpyKind kind);

/**
 * Allocates pinned memory for faster (and async) cuda memcopies
 */
std::shared_ptr<Buffer> createAlignedBufferCudaHost(
    size_t bufferSize,
    size_t sizeAlignment);

/**
 * When buffer->get() is null or its capacity is less than minSize,
 * a new buffer with pinned host memory is allocated.
 *
 * If buffer->get() is not null, the data in it's 'used' range will be copied
 * to the beginning of the new buffer, which is assigned to *buffer.
 */
void ensureMinCapacityAlignedCudaHost(
    std::shared_ptr<Buffer>* buffer,
    size_t minSize,
    size_t alignment);

void moveUsedToStartCuda(Buffer* buffer, cudaStream_t cudaStream);
#endif  // SDRTEST_SRC_CUDABUFFERS_H_
