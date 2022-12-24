//
// Created by Rick Kern on 10/29/22.
//

#ifndef SDRTEST_SRC_CUDAHOSTTODEVICEMEMCPY_H_
#define SDRTEST_SRC_CUDAHOSTTODEVICEMEMCPY_H_

#include <cuda_runtime.h>

#include <cstdint>

#include "Filter.h"

class CudaHostToDeviceMemcpy : public Filter {
 public:
  CudaHostToDeviceMemcpy(int32_t cudaDevice, cudaStream_t cudaStream);

  ~CudaHostToDeviceMemcpy() override = default;

  [[nodiscard]] std::shared_ptr<Buffer> requestBuffer(
      size_t port,
      size_t numBytes) override;
  void commitBuffer(size_t port, size_t numBytes) override;
  [[nodiscard]] size_t getOutputDataSize(size_t port) override;
  [[nodiscard]] size_t getOutputSizeAlignment(size_t port) override;
  void readOutput(
      const std::vector<std::shared_ptr<Buffer>>& portOutputs) override;

 private:
  int32_t mCudaDevice;
  cudaStream_t mCudaStream;
  std::shared_ptr<Buffer> mInputBuffer;
  bool mBufferCheckedOut;
};

#endif  // SDRTEST_SRC_CUDAHOSTTODEVICEMEMCPY_H_
