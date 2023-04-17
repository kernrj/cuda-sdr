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

#include "util.h"

using namespace std;

Result<vector<ImmutableRef<IBufferCopier>>> createOutputBufferCopierVector(
    IBufferCopier* copier,
    size_t outputPortCount) noexcept {
  try {
    vector<ImmutableRef<IBufferCopier>> copierVector;

    for (size_t i = 0; i < outputPortCount; i++) {
      copierVector.emplace_back(copier);
    }

    return makeValResult(std::move(copierVector));
  }
  IF_CATCH_RETURN_RESULT;
}

Result<std::string> stringPrintf(GS_FMT_STR(const char* fmt), ...) noexcept {
  try {
    va_list args;
    va_start(args, fmt);

    size_t stringSize = vsnprintf(nullptr, 0, fmt, args);
    std::string str(stringSize, 0);
    vsnprintf(str.data(), stringSize, fmt, args);

    va_end(args);

    return makeValResult(str);
  }
  IF_CATCH_RETURN_RESULT;
}
