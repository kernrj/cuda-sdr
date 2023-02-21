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

#ifndef GPUSDRPIPELINE_ADDCONSTFACTORY_H
#define GPUSDRPIPELINE_ADDCONSTFACTORY_H

#include "../AddConst.h"
#include "filters/FilterFactories.h"

class AddConstFactory final : public IAddConstFactory {
 public:
  explicit AddConstFactory(IFactories* factories)
      : mFactories(factories) {}
  Result<Filter> createAddConst(float addValue, int32_t cudaDevice, cudaStream_t cudaStream) noexcept final {
    return AddConst::create(addValue, cudaDevice, cudaStream, mFactories);
  }

 private:
  ConstRef<IFactories> mFactories;

  REF_COUNTED(AddConstFactory);
};

#endif  // GPUSDRPIPELINE_ADDCONSTFACTORY_H
