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

#ifndef GPUSDR_SRC_BASECUDAFILTER_H_
#define GPUSDR_SRC_BASECUDAFILTER_H_

#include <gpusdrpipeline/buffers/IBufferFactory.h>
#include <gpusdrpipeline/buffers/IBufferSliceFactory.h>
#include <gpusdrpipeline/buffers/IRelocatableResizableBufferFactory.h>
#include <gpusdrpipeline/filters/BaseSink.h>
#include <gpusdrpipeline/filters/Filter.h>

#include <cstdint>
#include <vector>

/**
 * Provides requestBuffer() and commitBuffer() methods. requestBuffer() provides GPU memory, with size and address
 * aligned to the value passed into the constructor via PortSpec.
 */
class BaseFilter : public virtual Filter, public BaseSink {
 public:
  BaseFilter() = delete;

 protected:
  BaseFilter(
      IRelocatableResizableBufferFactory* relocatableResizableBufferFactory,
      IBufferSliceFactory* slicedBufferFactory,
      size_t inputPortCount,
      IMemSet* memSet = nullptr) noexcept;

  ~BaseFilter() override = default;
};

#endif  // GPUSDR_SRC_BASECUDAFILTER_H_
