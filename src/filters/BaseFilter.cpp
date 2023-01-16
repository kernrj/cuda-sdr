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

#include "filters/BaseFilter.h"

using namespace std;

BaseFilter::BaseFilter(
    const std::shared_ptr<IRelocatableResizableBufferFactory>& relocatableResizableBufferFactory,
    const std::shared_ptr<IBufferSliceFactory>& slicedBufferFactory,
    size_t inputPortCount,
    const std::shared_ptr<IMemSet>& memSet)
    : BaseSink(relocatableResizableBufferFactory, slicedBufferFactory, inputPortCount, memSet) {}
