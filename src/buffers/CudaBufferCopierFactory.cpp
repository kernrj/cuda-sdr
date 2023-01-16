//
// Created by Rick Kern on 1/4/23.
//

#include "CudaBufferCopierFactory.h"

#include "CudaBufferCopier.h"

using namespace std;

std::shared_ptr<IBufferCopier> CudaBufferCopierFactory::createBufferCopier(
    int32_t cudaDevice,
    cudaStream_t cudaStream,
    cudaMemcpyKind memcpyKind) {
  return make_shared<CudaBufferCopier>(cudaDevice, cudaStream, memcpyKind);
}
