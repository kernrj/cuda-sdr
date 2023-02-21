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

#ifndef GPUSDRPIPELINE_FILTERDRIVERFACTORY_H
#define GPUSDRPIPELINE_FILTERDRIVERFACTORY_H

#include "FilterDriver.h"
#include "driver/IFilterDriverFactory.h"

class FilterDriverFactory final : public IFilterDriverFactory {
 public:
  explicit FilterDriverFactory(IFactories* factories)
      : mFactories(factories) {}

  Result<IFilterDriver> createFilterDriver() noexcept final {
    ISteppingDriverFactory* steppingDriverFactory = mFactories->getSteppingDriverFactory();
    ISteppingDriver* steppingDriver;
    UNWRAP_OR_FWD_RESULT(steppingDriver, steppingDriverFactory->createSteppingDriver());

    return makeRefResultNonNull<IFilterDriver>(new (std::nothrow) FilterDriver(steppingDriver));
  }

 private:
  ConstRef<IFactories> mFactories;

  REF_COUNTED(FilterDriverFactory);
};

#endif  // GPUSDRPIPELINE_FILTERDRIVERFACTORY_H
