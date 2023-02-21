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

#include "SysMemBufferCopier.h"

#include <cstring>

#include "GSLog.h"

Status SysMemBufferCopier::copy(void* dst, const void* src, size_t length) const noexcept {
  if (length > 0 && (dst == nullptr || src == nullptr)) {
    gslog(
        GSLOG_ERROR,
        "Cannot copy system memory: the source [%p] and destination [%p] must be non-null when length [%zu] > 0",
        src,
        dst,
        length);

    return Status_InvalidArgument;
  }

  memcpy(dst, src, length);

  return Status_Success;
}
