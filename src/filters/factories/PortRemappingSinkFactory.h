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

#ifndef GPUSDRPIPELINE_PORTREMAPPINGSINKFACTORY_H
#define GPUSDRPIPELINE_PORTREMAPPINGSINKFACTORY_H

#include "filters/FilterFactories.h"
#include "filters/PortRemappingSink.h"

class PortRemappingSinkFactory final : public IPortRemappingSinkFactory {
 public:
  Result<IPortRemappingSink> create() noexcept final {
    return makeRefResultNonNull<IPortRemappingSink>(new (std::nothrow) PortRemappingSink());
  }

  REF_COUNTED(PortRemappingSinkFactory);
};

#endif  // GPUSDRPIPELINE_PORTREMAPPINGSINKFACTORY_H
