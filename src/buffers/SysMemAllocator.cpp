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

#include "SysMemAllocator.h"

#include <new>

#include "Memory.h"

using namespace std;

Result<IMemory> SysMemAllocator::allocate(size_t size) noexcept {
  auto data = new (nothrow) uint8_t[size];
  NON_NULL_OR_RET(data);

  return makeRefResultNonNull<IMemory>(new (nothrow) Memory(data, size, freeMem, nullptr));
}

void SysMemAllocator::freeMem(uint8_t* data, [[maybe_unused]] void* context) noexcept {
  delete[] data;
}
