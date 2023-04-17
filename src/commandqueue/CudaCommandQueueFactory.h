//
// Created by Rick Kern on 4/1/23.
//

#ifndef GPUSDRPIPELINE_CUDACOMMANDQUEUEFACTORY_H
#define GPUSDRPIPELINE_CUDACOMMANDQUEUEFACTORY_H

#include "CudaCommandQueue.h"
#include "commandqueue/ICudaCommandQueueFactory.h"
#include "util/CudaDevicePushPop.h"

class CudaCommandQueueFactory final : public ICudaCommandQueueFactory {
 public:
  Result<ICudaCommandQueue> create(int32_t cudaDevice) noexcept final {
    CUDA_DEV_PUSH_POP_OR_RET_RESULT(cudaDevice);
    cudaStream_t stream = nullptr;
    SAFE_CUDA_OR_RET_RESULT(cudaStreamCreate(&stream));

    return CudaCommandQueue::create(cudaDevice, stream);
  }

  REF_COUNTED(CudaCommandQueueFactory);
};

#endif  // GPUSDRPIPELINE_CUDACOMMANDQUEUEFACTORY_H
