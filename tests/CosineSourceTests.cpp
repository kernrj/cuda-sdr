#include <cuComplex.h>
#include <cuda.h>
#include <gpusdrpipeline/Factories.h>
#include <gtest/gtest.h>

using namespace std;

TEST(WhenCosineSourceOutputs, ItMatchesCpuCalculations) {
  cudaStream_t cudaStream = nullptr;
  ConstRef<IFactories> factories = unwrap(getFactoriesSingleton());

  const size_t sampleRate = 100;
  const float frequency = 1.0f;

  ConstRef<ICudaCommandQueue> commandQueue = unwrap(factories->getCudaCommandQueueFactory()->create(0));
  const auto cosineSource =
      unwrap(factories->getCosineSourceFactory()
                 ->createCosineSource(SampleType_FloatComplex, sampleRate, frequency, commandQueue));
  auto hostToDeviceMemCopier = unwrap(
      factories->getCudaBufferCopierFactory()->createBufferCopier(commandQueue, cudaMemcpyHostToDevice));
  auto deviceToHostMemCopier = unwrap(
      factories->getCudaBufferCopierFactory()->createBufferCopier(commandQueue, cudaMemcpyDeviceToHost));
  auto cudaMemSet = unwrap(factories->getCudaMemSetFactory()->create(commandQueue));
  const auto cudaHostAllocator =
      unwrap(factories->getCudaAllocatorFactory()->createCudaAllocator(commandQueue, 32, true));
  const auto cudaGpuAllocator =
      unwrap(factories->getCudaAllocatorFactory()->createCudaAllocator(commandQueue, 32, false));
  const auto cudaGpuBufferFactory = unwrap(factories->createBufferFactory(cudaGpuAllocator));
  const auto cudaHostBufferFactory = unwrap(factories->createBufferFactory(cudaHostAllocator));
  vector<IBuffer*> outputBuffers;
  const size_t outputValueCount = sampleRate + 1;
  const size_t outputBufferSize = outputValueCount * sizeof(cuComplex);
  ConstRef<IBuffer> outputBuffer = unwrap(cudaGpuBufferFactory->createBuffer(outputBufferSize));
  outputBuffers.push_back(outputBuffer.get());
  THROW_IF_ERR(cudaMemSet->memSet(outputBuffers[0]->writePtr(), 0, outputBuffers[0]->range()->capacity()));
  THROW_IF_ERR(cosineSource->readOutput(outputBuffers.data(), 1));

  auto hostMem = unwrap(cudaHostBufferFactory->createBuffer(outputBuffers[0]->range()->used()));
  hostMem->range()->clearRange();

  THROW_IF_ERR(
      deviceToHostMemCopier->copy(hostMem->writePtr(), outputBuffers[0]->readPtr(), outputBuffers[0]->range()->used()));
  THROW_IF_ERR(hostMem->range()->setUsedRange(0, outputBuffers[0]->range()->used()));

  auto values = hostMem->readPtr<cuComplex>();

  cudaStreamSynchronize(cudaStream);

  const float maxError = 0.0001f;

  for (size_t i = 0; i < outputValueCount; i++) {
    const float theta = static_cast<float>(i) * frequency / sampleRate * M_PIf * 2.0f;
    ASSERT_TRUE(abs(values[i].x - cos(theta)) < maxError);
    ASSERT_TRUE(abs(values[i].y - sin(theta)) < maxError);
  }
}
