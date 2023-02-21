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

#ifndef GPUSDRPIPELINE_INT8TOFLOATFACTORY_H
#define GPUSDRPIPELINE_INT8TOFLOATFACTORY_H

#include "../Int8ToFloat.h"
#include "Factories.h"
#include "filters/FilterFactories.h"

class Int8ToFloatFactory final : public ICudaFilterFactory {
 public:
  explicit Int8ToFloatFactory(IFactories* factories) noexcept
      : mFactories(factories) {}

  Result<Filter> createFilter(int32_t cudaDevice, cudaStream_t cudaStream) noexcept final {
    return Int8ToFloat::create(cudaDevice, cudaStream, mFactories);
  }

 private:
  ConstRef<IFactories> mFactories;

  REF_COUNTED(Int8ToFloatFactory);
};

#endif  // GPUSDRPIPELINE_INT8TOFLOATFACTORY_H
