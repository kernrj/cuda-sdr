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

#include "util/Window.h"

#include <cmath>

using namespace std;

vector<float> createHammingWindow(size_t length) {
  vector<float> hammingWindow(length);

  for (size_t i = 0; i < length; i++) {
    hammingWindow[i] = 0.54f - 0.46f * cosf(i * 360.0f / (length - 1.0f));
  }

  return hammingWindow;
}