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

#include "util/CudaUtil.h"

#include <cuda_runtime.h>

#include "CudaErrors.h"

GS_C_LINKAGE Result<int32_t> gsGetCurrentCudaDevice() noexcept {
  int32_t device = -1;
  SAFE_CUDA_OR_RET_RESULT(cudaGetDevice(&device));

  return makeValResult<int32_t>(device);
}
