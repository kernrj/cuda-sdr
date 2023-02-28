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

#ifndef GPUSDR_IBUFFERUTIL_H
#define GPUSDR_IBUFFERUTIL_H

#include <gpusdrpipeline/buffers/IBuffer.h>
#include <gpusdrpipeline/buffers/IBufferCopier.h>

class IBufferUtil : public virtual IRef {
 public:
  [[nodiscard]] virtual Status appendToBuffer(
      IBuffer* buffer,
      const void* src,
      size_t count,
      const IBufferCopier* bufferCopier) const noexcept = 0;

  [[nodiscard]] virtual Status readFromBuffer(
      void* dst,
      IBuffer* buffer,
      size_t count,
      const IBufferCopier* bufferCopier) const noexcept = 0;

  [[nodiscard]] virtual Status moveFromBuffer(
      IBuffer* dst,
      IBuffer* src,
      size_t count,
      const IBufferCopier* bufferCopier) const noexcept = 0;

  ABSTRACT_IREF(IBufferUtil);
};

#endif  // GPUSDR_IBUFFERUTIL_H
