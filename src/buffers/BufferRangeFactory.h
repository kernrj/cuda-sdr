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

#ifndef GPUSDR_BUFFERRANGEFACTORY_H
#define GPUSDR_BUFFERRANGEFACTORY_H

#include "buffers/IBufferRangeFactory.h"

class BufferRangeFactory final : public IBufferRangeFactory {
 public:
  [[nodiscard]] Result<IBufferRangeMutableCapacity> createBufferRange() const noexcept final;

  REF_COUNTED(BufferRangeFactory);
};

#endif  // GPUSDR_BUFFERRANGEFACTORY_H
