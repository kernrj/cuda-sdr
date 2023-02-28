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

#ifndef GPUSDRPIPELINE_CUDAUTIL_H
#define GPUSDRPIPELINE_CUDAUTIL_H

#include <cuda_runtime.h>
#include <gpusdrpipeline/Result.h>

#include <cstdint>

GS_EXPORT [[nodiscard]] Result<int32_t> gsGetCurrentCudaDevice() noexcept;

[[nodiscard]] inline const char* cudaMemcpyKindName(cudaMemcpyKind memcpyKind) noexcept {
  switch (memcpyKind) {
    case cudaMemcpyHostToHost:
      return "host -> host";
    case cudaMemcpyHostToDevice:
      return "host -> device";
    case cudaMemcpyDeviceToDevice:
      return "device -> device";
    case cudaMemcpyDeviceToHost:
      return "device -> host";
    case cudaMemcpyDefault:
      return "default";
    default:
      return "unknown";
  }
}

#endif  // GPUSDRPIPELINE_CUDAUTIL_H
