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

#ifndef GPUSDRPIPELINE_CUDAMEMSETFACTORY_H
#define GPUSDRPIPELINE_CUDAMEMSETFACTORY_H

#include "CudaMemSet.h"
#include "buffers/ICudaMemSetFactory.h"

class CudaMemSetFactory final : public ICudaMemSetFactory {
 public:
  Result<IMemSet> create(ICudaCommandQueue* commandQueue) noexcept final {
    return makeRefResultNonNull<IMemSet>(new (std::nothrow) CudaMemSet(commandQueue));
  }

  REF_COUNTED(CudaMemSetFactory);
};

#endif  // GPUSDRPIPELINE_CUDAMEMSETFACTORY_H
