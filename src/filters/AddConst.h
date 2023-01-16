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

class AddConst : public BaseFilter {
 public:
  AddConst(float addValueToMagnitude, int32_t cudaDevice, cudaStream_t cudaStream, IFactories* factories);
  ~AddConst() override = default;

  [[nodiscard]] size_t getOutputDataSize(size_t port) override;
  [[nodiscard]] size_t getOutputSizeAlignment(size_t port) override;
  void readOutput(const std::vector<std::shared_ptr<IBuffer>>& portOutputs) override;

 private:
  static const size_t mAlignment;
  float mAddConst;
  int32_t mCudaDevice;
  cudaStream_t mCudaStream;

 private:
  [[nodiscard]] size_t getAvailableNumInputElements() const;
};

#endif  // SDRTEST_SRC_ADDCONST_H_
