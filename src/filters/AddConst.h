/*
 * Copyright 2022-2023 Rick Kern <kernrj@gmail.com>
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

#ifndef SDRTEST_SRC_ADDCONST_H_
#define SDRTEST_SRC_ADDCONST_H_

#include <cuda_runtime.h>

#include "Factories.h"
#include "buffers/IBufferSliceFactory.h"
#include "buffers/IRelocatableResizableBufferFactory.h"
#include "filters/BaseFilter.h"

class AddConst final : public BaseFilter {
 public:
  static Result<Filter> create(
      float addValueToMagnitude,
      ICudaCommandQueue* commandQueue,
      IFactories* factories) noexcept;

  [[nodiscard]] size_t getOutputDataSize(size_t port) noexcept final;
  [[nodiscard]] size_t getOutputSizeAlignment(size_t port) noexcept final;
  Status readOutput(IBuffer** portOutputBuffers, size_t numPorts) noexcept final;
  [[nodiscard]] size_t preferredInputBufferSize(size_t port) noexcept final;

 private:
  static const size_t mAlignment;
  float mAddConst;
  ConstRef<ICudaCommandQueue> mCommandQueue;

 private:
  [[nodiscard]] size_t getAvailableNumInputElements() const noexcept;

 private:
  AddConst(
      float addConst,
      ICudaCommandQueue* commandQueue,
      IRelocatableResizableBufferFactory* relocatableBufferFactory,
      IBufferSliceFactory* bufferSliceFactory,
      IMemSet* memSet,
      std::vector<ImmutableRef<IBufferCopier>>&& portOutputCopiers) noexcept;

  REF_COUNTED(AddConst);
};

#endif  // SDRTEST_SRC_ADDCONST_H_
