/*
 * Copyright 2022-2023 Rick Kern <kernrj@gmail.com>
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

#ifndef GPUSDRPIPELINE_UTIL_H
#define GPUSDRPIPELINE_UTIL_H

#include <vector>

#include "Result.h"
#include "buffers/IBufferCopier.h"

Result<std::vector<ImmutableRef<IBufferCopier>>> createOutputBufferCopierVector(
    IBufferCopier* copier,
    size_t outputPortCount = 1) noexcept;

GS_FMT_ATTR(1, 2) Result<std::string> stringPrintf(GS_FMT_STR(const char* fmt), ...) noexcept;

#endif  // GPUSDRPIPELINE_UTIL_H
