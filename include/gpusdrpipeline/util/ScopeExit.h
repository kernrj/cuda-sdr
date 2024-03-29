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

#ifndef SDRTEST_SRC_SCOPEEXIT_H_
#define SDRTEST_SRC_SCOPEEXIT_H_

#include <functional>

class ScopeExit {
 public:
  explicit ScopeExit(std::function<void()>&& doAtScopeExit) noexcept
      : mDoAtScopeExit(std::move(doAtScopeExit)) {}

  ~ScopeExit() {
    if (mDoAtScopeExit) {
      mDoAtScopeExit();
    }
  }

 private:
  const std::function<void()> mDoAtScopeExit;
};
#endif  // SDRTEST_SRC_SCOPEEXIT_H_
