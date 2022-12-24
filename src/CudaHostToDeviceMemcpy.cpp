//
// Created by Rick Kern on 10/29/22.
//

#include "CudaHostToDeviceMemcpy.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "CudaBuffers.h"
#include "CudaDevicePushPop.h"

using namespace std;

CudaHostToDeviceMemcpy::CudaHostToDeviceMemcpy(
    int32_t cudaDevice,
    cudaStream_t cudaStream)
    : mCudaDevice(cudaDevice), mCudaStream(cudaStream),
      mBufferCheckedOut(false) {}

shared_ptr<Buffer> CudaHostToDeviceMemcpy::requestBuffer(
    size_t port,
    size_t numBytes) {
  if (port >= 1) {
    throw runtime_error("Port [" + to_string(port) + "] is out of range");
  }

  if (mBufferCheckedOut) {
    throw runtime_error("Cannot request buffer - it is already checked out");
  }

  CudaDevicePushPop setAndRestore(mCudaDevice);
  ensureMinCapacity(&mInputBuffer, numBytes);

  return mInputBuffer->sliceRemaining();
}

void CudaHostToDeviceMemcpy::commitBuffer(size_t port, size_t numBytes) {
  if (port >= 1) {
    throw runtime_error("Port [" + to_string(port) + "] is out of range");
  }

  if (!mBufferCheckedOut) {
    throw runtime_error("Buffer cannot be committed - it was not checked out");
  }

  mInputBuffer->increaseEndOffset(numBytes);
  mBufferCheckedOut = false;
}

size_t CudaHostToDeviceMemcpy::getOutputDataSize(size_t port) {
  return mInputBuffer->used();
}

size_t CudaHostToDeviceMemcpy::getOutputSizeAlignment(size_t port) { return 1; }

void CudaHostToDeviceMemcpy::readOutput(
    const vector<shared_ptr<Buffer>>& portOutputs) {
  if (portOutputs.empty()) {
    throw runtime_error("One output port is required");
  }

  const auto& outBuffer = portOutputs[0];
  size_t copyNumBytes = min(outBuffer->remaining(), mInputBuffer->used());

  moveFromBufferCuda(
      outBuffer.get(),
      mInputBuffer.get(),
      copyNumBytes,
      mCudaStream,
      cudaMemcpyHostToDevice);

  moveUsedToStart(mInputBuffer.get());
}
