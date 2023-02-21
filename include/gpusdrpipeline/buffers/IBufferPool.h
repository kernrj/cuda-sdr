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

#ifndef GPUSDR_IBUFFERPOOL_H
#define GPUSDR_IBUFFERPOOL_H

#include <gpusdrpipeline/buffers/IBuffer.h>

#include <optional>

/**
 *
 */
class IBufferPool : public virtual IRef {
 public:
  /**
   * @return The number of bytes available in each buffer returned from the pool.
   */
  [[nodiscard]] virtual size_t getBufferSize() const noexcept = 0;

  /**
   * Gets a buffer. This will only be null if memory allocation failed.
   * This method may block - implementations may limit the number of buffers in a pool.
   */
  [[nodiscard]] virtual Result<IBuffer> getBuffer() noexcept = 0;

  /**
   * Sets *bufferOut to an IBuffer, provided one is immediately available or can be immediately created.
   * This method is non-blocking.
   *
   * @return 0 If *bufferOut was set, and an errno otherwise.
   */
  [[nodiscard]] virtual Result<IBuffer> tryGetBuffer() noexcept = 0;

  ABSTRACT_IREF(IBufferPool);
};
#endif  // GPUSDR_IBUFFERPOOL_H
