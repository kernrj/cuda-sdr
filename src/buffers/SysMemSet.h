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

#ifndef GPUSDRPIPELINE_SYSMEMSET_H
#define GPUSDRPIPELINE_SYSMEMSET_H

#include <cstring>

#include "buffers/IMemSet.h"

class SysMemSet final : public IMemSet {
 public:
  Status memSet(void* data, uint8_t value, size_t byteCount) noexcept final {
    ::memset(data, value, byteCount);
    return Status_Success;
  }

  REF_COUNTED(SysMemSet);
};

#endif  // GPUSDRPIPELINE_SYSMEMSET_H
