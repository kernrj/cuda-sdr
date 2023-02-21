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

#ifndef GPUSDRPIPELINE_STEPPINGDRIVERFACTORY_H
#define GPUSDRPIPELINE_STEPPINGDRIVERFACTORY_H

#include "driver/ISteppingDriverFactory.h"
#include "driver/SteppingDriver.h"

class SteppingDriverFactory final : public ISteppingDriverFactory {
 public:
  Result<ISteppingDriver> createSteppingDriver() noexcept final {
    return makeRefResultNonNull<ISteppingDriver>(new (std::nothrow) SteppingDriver());
  }

  REF_COUNTED(SteppingDriverFactory);
};

#endif  // GPUSDRPIPELINE_STEPPINGDRIVERFACTORY_H
