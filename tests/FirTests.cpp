#include <cuComplex.h>
#include <cuda.h>
#include <gpusdrpipeline/Factories.h>
#include <gpusdrpipeline/util/create.h>
#include <gtest/gtest.h>

#include <string>

using namespace std;

TEST(WhenThereAre4InputsWithDecimation2AndTwoInputCommits, ItProduces2CorrectOutputs) {
  const int32_t cudaDevice = 0;
  cudaStream_t cudaStream = nullptr;

  constexpr size_t tapCount = 2;
  vector<float> taps(tapCount);
  taps[0] = 0.5f;
  taps[1] = 1.0f;

  IFactories* factories = getFactoriesSingleton();
  auto hostToDeviceMemCopier =
      factories->getCudaBufferCopierFactory()->createBufferCopier(cudaDevice, cudaStream, cudaMemcpyHostToDevice);
  auto deviceToHostMemCopier =
      factories->getCudaBufferCopierFactory()->createBufferCopier(cudaDevice, cudaStream, cudaMemcpyDeviceToHost);
  auto cudaMemSet = factories->getCudaMemSetFactory()->create(cudaDevice, cudaStream);

  constexpr size_t decimation = 2;
  shared_ptr<Filter> fir =
      factories->getFirFactory()->createFirCF(decimation, taps.data(), taps.size(), cudaDevice, cudaStream);

  /*
   * Indices 0 and 1 are used to create the first output.
   * Indices 1 and 2 are not used in the calculation of an output because decimation = 2 (every other output skipped).
   * Indices 2 and 3 are used to create the second output.
   * Indices 3 and 4 are not used (decimation).
   */
  cuComplex cpuData1[] = {
      make_cuComplex(0.1f, 0.2f),
      make_cuComplex(0.3f, 0.4f),
      make_cuComplex(0.5f, 0.6f),
  };
  cuComplex cpuData2[] = {
      make_cuComplex(0.7f, 0.8f),
      make_cuComplex(0.9f, 0.9f),
  };

  const size_t inputDataCount = 4;
  const size_t expectedOutputCount = 2;

  auto buffer = fir->requestBuffer(0, inputDataCount * sizeof(cuComplex));
  auto* gpuData = buffer->writePtr<cuComplex>();
  hostToDeviceMemCopier->copy(gpuData, cpuData1, sizeof(cpuData1));
  fir->commitBuffer(0, sizeof(cpuData1));

  buffer = fir->requestBuffer(0, sizeof(cpuData2));
  gpuData = buffer->writePtr<cuComplex>();
  hostToDeviceMemCopier->copy(gpuData, cpuData2, sizeof(cpuData2));
  fir->commitBuffer(0, sizeof(cpuData2));

  const auto cudaAllocator = factories->getCudaAllocatorFactory()->createCudaAllocator(0, nullptr, 32, false);
  const auto cudaBufferFactory = factories->getBufferFactory(cudaAllocator);
  vector<shared_ptr<IBuffer>> outputBuffers;
  const size_t outputBufferSize = 2 * fir->getOutputDataSize(0);  // create oversized buffer, expect not-full on output.
  outputBuffers.push_back(cudaBufferFactory->createBuffer(outputBufferSize));
  cudaMemSet->memSet(outputBuffers[0]->writePtr(), 0, outputBuffers[0]->range()->capacity());

  fir->readOutput(outputBuffers);

  auto& outputBuffer = outputBuffers[0];

  ASSERT_EQ(expectedOutputCount * sizeof(cuComplex), outputBuffer->range()->used());

  cuComplex hostOutputBuffer[expectedOutputCount];
  deviceToHostMemCopier->copy(hostOutputBuffer, outputBuffer->readPtr(), expectedOutputCount * sizeof(cuComplex));

  cuComplex expectedValues[expectedOutputCount] = {
      make_cuComplex(0.35f, 0.5f),
      make_cuComplex(0.95f, 1.1f),
  };

  for (size_t i = 0; i < expectedOutputCount; i++) {
    ASSERT_TRUE(abs(expectedValues[i].x - hostOutputBuffer[i].x) < 0.001) << "real-value at index [" << i << "]";
    ASSERT_TRUE(abs(expectedValues[i].y - hostOutputBuffer[i].y) < 0.001) << "imaginary-value at index [" << i << "]";
  }
}

TEST(WhenTheFirstReadCantFitAllAvailableInputs, ItDoesntSkipAnyInputValues) {
  const int32_t cudaDevice = 0;
  cudaStream_t cudaStream = nullptr;

  constexpr size_t tapCount = 3;
  vector<float> taps(tapCount);
  taps[0] = 0.5f;
  taps[1] = 1.0f;
  taps[2] = 0.25f;

  IFactories* factories = getFactoriesSingleton();
  auto firCcfFactory = factories->getFirFactory();
  constexpr size_t decimation = 2;
  shared_ptr<Filter> fir =
      factories->getFirFactory()->createFirCF(decimation, taps.data(), taps.size(), cudaDevice, cudaStream);

  auto hostToDeviceMemCopier =
      factories->getCudaBufferCopierFactory()->createBufferCopier(cudaDevice, cudaStream, cudaMemcpyHostToDevice);
  auto deviceToHostMemCopier =
      factories->getCudaBufferCopierFactory()->createBufferCopier(cudaDevice, cudaStream, cudaMemcpyDeviceToHost);
  auto bufferSliceFactory = factories->getBufferSliceFactory();
  auto cudaAllocator = factories->getCudaAllocatorFactory()
                           ->createCudaAllocator(cudaDevice, cudaStream, fir->getOutputSizeAlignment(0), false);
  auto cudaBufferFactory = factories->getBufferFactory(cudaAllocator);
  auto cudaMemSet = factories->getCudaMemSetFactory()->create(cudaDevice, cudaStream);

  cuComplex cpuData[] = {
      make_cuComplex(0.1f, 0.2f),
      make_cuComplex(0.3f, 0.4f),
      make_cuComplex(0.5f, 0.6f),
      make_cuComplex(0.7f, 0.8f),
      make_cuComplex(0.1f, 0.2f),
      make_cuComplex(0.3f, 0.4f),
      make_cuComplex(0.5f, 0.6f),
      make_cuComplex(0.7f, 0.8f),
  };

  /*
   * (0.1f, 0.2f), \
   * (0.3f, 0.4f), | - output 0        \
   * (0.5f, 0.6f), /   \               | - skipped (because decimation = 2)
   * (0.7f, 0.8f),     | - output 1    /    \
   * (0.1f, 0.2f), \  /                     | - skipped (decimation)
   * (0.3f, 0.4f), | - output 2       \    /
   * (0.5f, 0.6f),/                   | - skipped (decimation)
   * (0.7f, 0.8f),                   /
   */

  const size_t expectedOutputCount1 = 1;
  const size_t expectedOutputCount2 = 2;

  auto buffer = fir->requestBuffer(0, sizeof(cpuData));
  auto* gpuData = buffer->writePtr<cuComplex>();
  hostToDeviceMemCopier->copy(gpuData, cpuData, sizeof(cpuData));
  fir->commitBuffer(0, sizeof(cpuData));

  const size_t outputBuffer1Size =
      expectedOutputCount1 * sizeof(cuComplex);  // Create a buffer containing 1 element, so we need to call twice.
  const size_t outputBuffer2Size =
      expectedOutputCount2 * sizeof(cuComplex);  // Create a buffer containing 1 element, so we need to call twice.
  vector<shared_ptr<IBuffer>> outputBuffers1;
  vector<shared_ptr<IBuffer>> outputBuffers2;

  outputBuffers1.push_back(cudaBufferFactory->createBuffer(outputBuffer1Size));
  outputBuffers2.push_back(cudaBufferFactory->createBuffer(outputBuffer2Size));

  cudaMemSet->memSet(outputBuffers1[0]->writePtr(), 0, outputBuffers1[0]->range()->capacity());
  cudaMemSet->memSet(outputBuffers2[0]->writePtr(), 0, outputBuffers1[0]->range()->capacity());

  // Alignment causes capacity to be greater
  outputBuffers1[0] = bufferSliceFactory->slice(outputBuffers1[0], 0, outputBuffer1Size);
  outputBuffers2[0] = bufferSliceFactory->slice(outputBuffers2[0], 0, outputBuffer2Size);

  outputBuffers1[0]->range()->clearRange();  // Slice sets the range to [0, outputBuffer1Size), clear it to [0, 0).
  outputBuffers2[0]->range()->clearRange();

  fir->readOutput(outputBuffers1);
  fir->readOutput(outputBuffers2);

  auto& outputBuffer1 = outputBuffers1[0];
  auto& outputBuffer2 = outputBuffers2[0];

  ASSERT_EQ(expectedOutputCount1 * sizeof(cuComplex), outputBuffer1->range()->used());
  ASSERT_EQ(expectedOutputCount2 * sizeof(cuComplex), outputBuffer2->range()->used());

  cuComplex hostOutputBuffer1[expectedOutputCount1];
  cuComplex hostOutputBuffer2[expectedOutputCount2];

  deviceToHostMemCopier->copy(hostOutputBuffer1, outputBuffer1->readPtr(), expectedOutputCount1 * sizeof(cuComplex));
  deviceToHostMemCopier->copy(hostOutputBuffer2, outputBuffer2->readPtr(), expectedOutputCount2 * sizeof(cuComplex));

  cuComplex expectedValues1[expectedOutputCount1] = {
      make_cuComplex(0.475f, 0.65f),
  };
  cuComplex expectedValues2[expectedOutputCount2] = {
      make_cuComplex(0.975f, 1.15f),
      make_cuComplex(0.475f, 0.65f),
  };

  for (size_t i = 0; i < expectedOutputCount1; i++) {
    ASSERT_TRUE(abs(expectedValues1[i].x - hostOutputBuffer1[i].x) < 0.001f)
        << "Expected real-value [" << expectedValues1[i].x << "] but got [" << hostOutputBuffer1[i].x << "] at index ["
        << i << "]";
    ASSERT_TRUE(abs(expectedValues1[i].y - hostOutputBuffer1[i].y) < 0.001f)
        << "Expected imaginary-value [" << expectedValues1[i].y << "] but got [" << hostOutputBuffer1[i].y
        << "] at index [" << i << "]";
  }

  for (size_t i = 0; i < expectedOutputCount2; i++) {
    ASSERT_TRUE(abs(expectedValues2[i].x - hostOutputBuffer2[i].x) < 0.001f)
        << "Expected real-value [" << expectedValues2[i].x << "] but got [" << hostOutputBuffer2[i].x << "] at index ["
        << i << "]";
    ASSERT_TRUE(abs(expectedValues2[i].y - hostOutputBuffer2[i].y) < 0.001f)
        << "Expected imaginary-value [" << expectedValues2[i].y << "] but got [" << hostOutputBuffer2[i].y
        << "] at index [" << i << "]";
  }
}
