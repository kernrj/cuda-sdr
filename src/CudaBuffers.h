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
 * Creates an aligned buffer, and causes a GPU sync when allocating, and another
 * when the returned buffer is freed.
 *
 * @param elementCount The number of elements in the returned buffer.
 */
OwnedBuffer createAlignedBuffer(
    size_t bufferSize,
    size_t sizeAlignment,
    cudaStream_t cudaStream = nullptr);

/**
 * Creates an aligned buffer, and causes a GPU sync when allocating, and another
 * when the returned buffer is freed.
 *
 * @param elementCount The number of elements in the returned buffer.
 */
void ensureMinCapacityAligned(
    OwnedBuffer* buffer,
    size_t minSize,
    size_t alignment,
    cudaStream_t cudaStream = nullptr);

#endif  // SDRTEST_SRC_CUDABUFFERS_H_
