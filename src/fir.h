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

#ifndef SDRTEST_SRC_FIR_H_
#define SDRTEST_SRC_FIR_H_

#include <cuda_runtime.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "Buffer.h"
#include "Filter.h"

class FirCcf : public Filter {
 public:
  FirCcf(
      uint32_t decimation,
      const std::vector<float>& taps,
      int32_t cudaDevice,
      cudaStream_t cudaStream);

  [[nodiscard]] Buffer requestBuffer(size_t port, size_t numBytes) override;
  void commitBuffer(size_t port, size_t numBytes) override;
  [[nodiscard]] size_t getOutputDataSize(size_t port) override;
  [[nodiscard]] size_t getOutputSizeAlignment(size_t port) override;
  void readOutput(Buffer* portOutputs, size_t portOutputCount) override;

 private:
  static const size_t mAlignment;
  uint32_t mDecimation;
  OwnedBuffer mTaps;
  int32_t mTapCount;
  OwnedBuffer mBuffer;
  cudaStream_t mCudaStream;
  int32_t mCudaDevice;
  bool mBufferCheckedOut;

 private:
  void setTaps(const std::vector<float>& tapsReversed);
};

#endif  // SDRTEST_SRC_FIR_H_
