/*
 * Copyright 2023 Rick Kern <kernrj@gmail.com>
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

#include <cstddef>
#include <cstdint>
#include <memory>

#ifndef GPUSDR_IALLOCATOR_H
#define GPUSDR_IALLOCATOR_H

/**
 * Abstracts the creation and deletion of a memory region.
 */
class IAllocator {
 public:
  virtual ~IAllocator() = default;

  /**
   * Allocates memory at least [size] bytes long.
   *
   * @param size The minimum size of the buffer. Must be > 0. Some implementations may return larger buffers (e.g.
   *             padding for compatibility with vectorized operations).
   * @param sizeOut If non-null, this is set to the size of the allocated buffer.
   */
  virtual std::shared_ptr<uint8_t> allocate(size_t size, size_t* sizeOut = nullptr) = 0;
};

#endif  // GPUSDR_IALLOCATOR_H
