//
// Created by Rick Kern on 1/8/23.
//

#ifndef GPUSDRPIPELINE_FILTERFACTORIES_H
#define GPUSDRPIPELINE_FILTERFACTORIES_H

#include <cuda_runtime.h>
#include <gpusdrpipeline/filters/Filter.h>
#include <gpusdrpipeline/filters/IHackrfSource.h>

#include "FirType.h"

class ICudaMemcpyFilterFactory {
 public:
  virtual ~ICudaMemcpyFilterFactory() = default;
  virtual std::shared_ptr<Filter> createCudaMemcpy(
      cudaMemcpyKind memcpyKind,
      int32_t cudaDevice,
      cudaStream_t cudaStream) = 0;
};

class IAacFileWriterFactory {
 public:
  virtual ~IAacFileWriterFactory() = default;
  virtual std::shared_ptr<Sink> createAacFileWriter(
      const char* outputFileName,
      int32_t sampleRate,
      int32_t bitRate) = 0;
};

class IAddConstFactory {
 public:
  virtual ~IAddConstFactory() = default;
  virtual std::shared_ptr<Filter> createAddConst(
      float addValueToAmplitude,
      int32_t cudaDevice,
      cudaStream_t cudaStream) = 0;
};

class IAddConstToVectorLengthFactory {
 public:
  virtual ~IAddConstToVectorLengthFactory() = default;
  virtual std::shared_ptr<Filter> createAddConstToVectorLength(
      float addValueToMagnitude,
      int32_t cudaDevice,
      cudaStream_t cudaStream) = 0;
};

class ICosineSourceFactory {
 public:
  virtual ~ICosineSourceFactory() = default;
  virtual std::shared_ptr<Source> createCosineSource(
      float sampleRate,
      float frequency,
      int32_t cudaDevice,
      cudaStream_t cudaStream) = 0;
};

class IFileReaderFactory {
 public:
  virtual ~IFileReaderFactory() = default;
  virtual std::shared_ptr<Source> createFileReader(const char* fileName) = 0;
};

class IFirFactory {
 public:
  virtual ~IFirFactory() = default;
  virtual std::shared_ptr<Filter> createFir(
      FirType firType,
      size_t decimation,
      const float* taps,
      size_t tapCount,
      int32_t cudaDevice,
      cudaStream_t cudaStream) = 0;
};

class IHackrfSourceFactory {
 public:
  virtual ~IHackrfSourceFactory() = default;
  virtual std::shared_ptr<IHackrfSource> createHackrfSource(
      int32_t deviceIndex,
      uint64_t frequency,
      double sampleRate,
      size_t maxBufferCountBeforeDropping) = 0;
};

class ICudaFilterFactory {
 public:
  virtual ~ICudaFilterFactory() = default;
  virtual std::shared_ptr<Filter> createFilter(int32_t cudaDevice, cudaStream_t cudaStream) = 0;
};

class IQuadDemodFactory {
 public:
  virtual ~IQuadDemodFactory() = default;

  virtual std::shared_ptr<Filter> create(float gain, int32_t cudaDevice, cudaStream_t cudaStream) = 0;
};

#endif  // GPUSDRPIPELINE_FILTERFACTORIES_H
