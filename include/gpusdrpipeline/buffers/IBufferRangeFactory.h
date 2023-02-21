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

#ifndef GPUSDR_IBUFFERRANGEFACTORY_H
#define GPUSDR_IBUFFERRANGEFACTORY_H

#include <gpusdrpipeline/Result.h>
#include <gpusdrpipeline/buffers/IBufferRangeMutableCapacity.h>

class IBufferRangeFactory : public virtual IRef {
 public:
  [[nodiscard]] virtual Result<IBufferRangeMutableCapacity> createBufferRange() const noexcept = 0;

  [[nodiscard]] Result<IBufferRangeMutableCapacity> createBufferRangeWithCapacity(size_t capacity) const {
    IBufferRangeMutableCapacity* range;
    UNWRAP_OR_FWD_RESULT(range, createBufferRange());

    range->setCapacity(capacity);

    return makeRefResultNonNull(range);
  }

 protected:
  ABSTRACT_IREF(IBufferRangeFactory);
};

#endif  // GPUSDR_IBUFFERRANGEFACTORY_H
