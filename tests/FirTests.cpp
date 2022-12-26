#include <cuComplex.h>
#include <cuda.h>
#include <gtest/gtest.h>

#include "fir.h"
#include "CudaBuffers.h"

using namespace std;

TEST(WhenSomething, ItDoesSomething) {
    constexpr size_t tapCount = 2;
    vector<float> taps(tapCount);
    taps[0] = 0.5f;
    taps[1] = 1.0f;

    constexpr size_t decimation = 2;
    FirCcf fir(decimation, taps, 0, nullptr);

    const size_t expectedOutputCount = 32;
    /*
     * Output must be at least 'alignment' elements long.
     * taps.size() - 1, because of how the FIR buffers samples.
     */
    const size_t dataCount = decimation * expectedOutputCount + tapCount - 1;

    cuComplex cpuData1[] = {
        make_cuComplex(0.1f, 0.2f),
        make_cuComplex(0.3f, 0.4f),
        make_cuComplex(0.5f, 0.6f),
    };
    cuComplex cpuData2[dataCount - sizeof(cpuData1) / sizeof(cpuData1[0])] = {
        make_cuComplex(0.7f, 0.8f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f)
    };

    auto buffer = fir.requestBuffer(0, dataCount * sizeof(cuComplex));
    cuComplex* gpuData = buffer->writePtr<cuComplex>();
    cudaMemcpy(gpuData, cpuData1, sizeof(cpuData1), cudaMemcpyHostToDevice);
    fir.commitBuffer(0, sizeof(cpuData1));

    buffer = fir.requestBuffer(0, sizeof(cpuData2));
    gpuData = buffer->writePtr<cuComplex>();
    cudaMemcpy(gpuData, cpuData2, sizeof(cpuData2), cudaMemcpyHostToDevice);
    fir.commitBuffer(0, sizeof(cpuData2));

    vector<shared_ptr<Buffer>> outputBuffers;
    outputBuffers.resize(1);
    ensureMinCapacityAlignedCuda(
        &outputBuffers[0],
        fir.getAlignedOutputDataSize(0),
        32 * sizeof(cuComplex),
        nullptr);
    cudaMemset(outputBuffers[0]->writePtr(), 0, outputBuffers[0]->capacity());

    fir.readOutput(outputBuffers);

    auto& outputBuffer = outputBuffers[0];

    ASSERT_EQ(expectedOutputCount * sizeof(cuComplex), outputBuffer->used());

    cuComplex hostOutputBuffer[expectedOutputCount];
    cudaMemcpy(hostOutputBuffer, outputBuffer->readPtr(), expectedOutputCount * sizeof(cuComplex), cudaMemcpyDeviceToHost);

    cuComplex expectedValues[expectedOutputCount] = {
        make_cuComplex(0.35f, 0.5f),
        make_cuComplex(0.95f, 1.1f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
        make_cuComplex(0.0f, 0.0f),
    };

    for(size_t i = 0; i < expectedOutputCount; i++) {
        ASSERT_TRUE(abs(expectedValues[i].x - hostOutputBuffer[i].x) < 0.001) << "real-value at index [" << i << "]";
        ASSERT_TRUE(abs(expectedValues[i].y - hostOutputBuffer[i].y) < 0.001) << "imaginary-value at index [" << i << "]";
    }
}
