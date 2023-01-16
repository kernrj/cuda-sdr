#include <cuComplex.h>
#include <cuda.h>
#include <gpusdrpipeline/Factories.h>
#include <gpusdrpipeline/util/create.h>
#include <gtest/gtest.h>

#include <string>

using namespace std;

TEST(WhenCosineSourceOutputs, ItMatchesCpuCalculations) {
  const int32_t cudaDevice = 0;
  cudaStream_t cudaStream = nullptr;
  IFactories* factories = getFactoriesSingleton();

  const size_t sampleRate = 100;
  const float frequency = 1.0f;

  const auto cosineSource =
      factories->getCosineSourceFactory()->createCosineSource(sampleRate, frequency, cudaDevice, cudaStream);
  auto hostToDeviceMemCopier =
      factories->getCudaBufferCopierFactory()->createBufferCopier(cudaDevice, cudaStream, cudaMemcpyHostToDevice);
  auto deviceToHostMemCopier =
      factories->getCudaBufferCopierFactory()->createBufferCopier(cudaDevice, cudaStream, cudaMemcpyDeviceToHost);
  auto cudaMemSet = factories->getCudaMemSetFactory()->create(cudaDevice, cudaStream);
  const auto cudaHostAllocator =
      factories->getCudaAllocatorFactory()->createCudaAllocator(cudaDevice, cudaStream, 32, true);
  const auto cudaGpuAllocator =
      factories->getCudaAllocatorFactory()->createCudaAllocator(cudaDevice, cudaStream, 32, false);
  const auto cudaGpuBufferFactory = factories->getBufferFactory(cudaGpuAllocator);
  const auto cudaHostBufferFactory = factories->getBufferFactory(cudaHostAllocator);
  vector<shared_ptr<IBuffer>> outputBuffers;
  const size_t outputValueCount = sampleRate + 1;
  const size_t outputBufferSize = outputValueCount * sizeof(cuComplex);
  outputBuffers.push_back(cudaGpuBufferFactory->createBuffer(outputBufferSize));
  cudaMemSet->memSet(outputBuffers[0]->writePtr(), 0, outputBuffers[0]->range()->capacity());
  cosineSource->readOutput(outputBuffers);

  auto hostMem = cudaHostBufferFactory->createBuffer(outputBuffers[0]->range()->used());
  hostMem->range()->clearRange();
  printf(
      "Copying [%zu] bytes from [%p] to [%p]\n",
      outputBuffers[0]->range()->used(),
      outputBuffers[0]->readPtr(),
      hostMem->writePtr());
  deviceToHostMemCopier->copy(hostMem->writePtr(), outputBuffers[0]->readPtr(), outputBuffers[0]->range()->used());
  hostMem->range()->setUsedRange(0, outputBuffers[0]->range()->used());

  auto values = hostMem->readPtr<cuComplex>();

  cudaStreamSynchronize(cudaStream);

  const float maxError = 0.0001f;

  for (size_t i = 0; i < outputValueCount; i++) {
    const float theta = static_cast<float>(i) * frequency / sampleRate * M_PIf * 2.0f;
    ASSERT_TRUE(abs(values[i].x - cos(theta)) < maxError);
    ASSERT_TRUE(abs(values[i].y - sin(theta)) < maxError);
  }
}
