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

#include <gpusdrpipeline/IMemory.h>
#include <gpusdrpipeline/IRef.h>
#include <gpusdrpipeline/Result.h>

#include <cstddef>
#include <cstdint>

#ifndef GPUSDR_IALLOCATOR_H
#define GPUSDR_IALLOCATOR_H

/**
 * Abstracts the creation and deletion of a memory region.
 */
class IAllocator : public virtual IRef {
 public:
  /**
   * Allocates memory at least [size] bytes long.
   *
   * @param size The minimum size of the buffer. Must be > 0. Some implementations may return larger buffers (e.g.
   *             padding for compatibility with vectorized operations).
   */
  [[nodiscard]] virtual Result<IMemory> allocate(size_t size) noexcept = 0;

  ABSTRACT_IREF(IAllocator);
};

#endif  // GPUSDR_IALLOCATOR_H
