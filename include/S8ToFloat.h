/*
 * Copyright 2022 Rick Kern <kernrj@gmail.com>
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

#ifndef SDRTEST_SRC_S8TOFLOAT_H_
#define SDRTEST_SRC_S8TOFLOAT_H_

#include "Buffer.h"
#include "Filter.h"

class CudaInt8ToFloat : public Filter {
 public:
  explicit CudaInt8ToFloat(int32_t cudaDevice, cudaStream_t cudaStream);

  [[nodiscard]] std::shared_ptr<Buffer> requestBuffer(size_t port, size_t numBytes) override;
  void commitBuffer(size_t port, size_t numBytes) override;
  [[nodiscard]] size_t getOutputDataSize(size_t port) override;
  [[nodiscard]] size_t getOutputSizeAlignment(size_t port) override;
  void readOutput(const std::vector<std::shared_ptr<Buffer>>& portOutputs) override;

 private:
  static const size_t mAlignment;
  int32_t mCudaDevice;
  cudaStream_t mCudaStream;
  std::shared_ptr<Buffer> mInputBuffer;
  bool mBufferCheckedOut;
};

#endif  // SDRTEST_SRC_S8TOFLOAT_H_
