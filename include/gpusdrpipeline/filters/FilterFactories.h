/*
 * Copyright 2023 Rick Kern <kernrj@gmail.com>
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

#ifndef GPUSDRPIPELINE_FILTERFACTORIES_H
#define GPUSDRPIPELINE_FILTERFACTORIES_H

#include <cuda_runtime.h>
#include <gpusdrpipeline/Modulation.h>
#include <gpusdrpipeline/SampleType.h>
#include <gpusdrpipeline/filters/Filter.h>
#include <gpusdrpipeline/filters/IHackrfSource.h>
#include <gpusdrpipeline/filters/IPortRemappingSink.h>
#include <gpusdrpipeline/filters/IReadByteCountMonitor.h>

GS_EXPORT [[nodiscard]] Result<Filter> createFilter(const char* name, const char* jsonParameters) noexcept;
GS_EXPORT [[nodiscard]] Result<Source> createSource(const char* name, const char* jsonParameters) noexcept;
GS_EXPORT [[nodiscard]] Result<Source> createSink(const char* name, const char* jsonParameters) noexcept;

GS_EXPORT [[nodiscard]] bool hasFilterFactory(const char* name) noexcept;
GS_EXPORT [[nodiscard]] bool hasSourceFactory(const char* name) noexcept;
GS_EXPORT [[nodiscard]] bool hasSinkFactory(const char* name) noexcept;

GS_EXPORT void registerFilterFactory(
    const char* name,
    void* context,
    Filter* (*filterFactory)(const char* jsonParamters)) noexcept;

GS_EXPORT void registerSourceFactory(
    const char* name,
    void* context,
    Source* (*filterFactory)(const char* jsonParamters)) noexcept;

GS_EXPORT void registerSinkFactory(
    const char* name,
    void* context,
    Sink* (*sinkFactory)(const char* jsonParamters)) noexcept;

class ICudaMemcpyFilterFactory : public virtual IRef {
 public:
  [[nodiscard]] virtual Result<Filter> createCudaMemcpy(
      cudaMemcpyKind memcpyKind,
      int32_t cudaDevice,
      cudaStream_t cudaStream) noexcept = 0;

  ABSTRACT_IREF(ICudaMemcpyFilterFactory);
};

class IAacFileWriterFactory : public virtual IRef {
 public:
  [[nodiscard]] virtual Result<Sink> createAacFileWriter(
      const char* outputFileName,
      int32_t sampleRate,
      int32_t bitRate,
      int32_t cudaDevice,
      cudaStream_t cudaStream) noexcept = 0;

  ABSTRACT_IREF(IAacFileWriterFactory);
};

class IAddConstFactory : public virtual IRef {
 public:
  [[nodiscard]] virtual Result<Filter> createAddConst(
      float addValueToAmplitude,
      int32_t cudaDevice,
      cudaStream_t cudaStream) noexcept = 0;

  ABSTRACT_IREF(IAddConstFactory);
};

class IAddConstToVectorLengthFactory : public virtual IRef {
 public:
  [[nodiscard]] virtual Result<Filter> createAddConstToVectorLength(
      float addValueToMagnitude,
      int32_t cudaDevice,
      cudaStream_t cudaStream) noexcept = 0;

  ABSTRACT_IREF(IAddConstToVectorLengthFactory);
};

class ICosineSourceFactory : public virtual IRef {
 public:
  [[nodiscard]] virtual Result<Source> createCosineSource(
      SampleType sampleType,
      float sampleRate,
      float frequency,
      int32_t cudaDevice,
      cudaStream_t cudaStream) noexcept = 0;

  ABSTRACT_IREF(ICosineSourceFactory);
};

class IFileReaderFactory : public virtual IRef {
 public:
  [[nodiscard]] virtual Result<Source> createFileReader(const char* fileName) noexcept = 0;

  ABSTRACT_IREF(IFileReaderFactory);
};

class IFirFactory : public virtual IRef {
 public:
  [[nodiscard]] virtual Result<Filter> createFir(
      SampleType tapType,
      SampleType elementType,
      size_t decimation,
      const float* taps,
      size_t tapCount,
      int32_t cudaDevice,
      cudaStream_t cudaStream) noexcept = 0;

  ABSTRACT_IREF(IFirFactory);
};

class IHackrfSourceFactory : public virtual IRef {
 public:
  [[nodiscard]] virtual Result<IHackrfSource> createHackrfSource(
      int32_t deviceIndex,
      uint64_t frequency,
      double sampleRate,
      size_t maxBufferCountBeforeDropping) noexcept = 0;

  ABSTRACT_IREF(IHackrfSourceFactory);
};

class ICudaFilterFactory : public virtual IRef {
 public:
  [[nodiscard]] virtual Result<Filter> createFilter(int32_t cudaDevice, cudaStream_t cudaStream) noexcept = 0;

  ABSTRACT_IREF(ICudaFilterFactory);
};

class IQuadDemodFactory : public virtual IRef {
 public:
  /**
   * @param fskDeviation Only used for FM
   */
  [[nodiscard]] virtual Result<Filter> create(
      Modulation modulation,
      float rfSampleRate,
      float fskDeviation,
      int32_t cudaDevice,
      cudaStream_t cudaStream) noexcept = 0;

  ABSTRACT_IREF(IQuadDemodFactory);
};

class IPortRemappingSinkFactory : public virtual IRef {
 public:
  [[nodiscard]] virtual Result<IPortRemappingSink> create(Sink* mapToSink) noexcept = 0;

  ABSTRACT_IREF(IPortRemappingSinkFactory);
};

class IRfToPcmAudioFactory : public virtual IRef {
 public:
  [[nodiscard]] virtual Result<Filter> create(
      float rfSampleRate,
      Modulation modulation,
      size_t rfLowPassDecim,
      size_t audioLowPassDecim,
      float centerFrequency,
      float channelFrequency,
      float channelWidth,
      float fskDevationIfFm,
      float rfLowPassDbAttenuation,
      float audioLowPassDbAttenuation,
      int32_t cudaDevice,
      cudaStream_t cudaStream) noexcept = 0;

  ABSTRACT_IREF(IRfToPcmAudioFactory);
};

class IReadByteCountMonitorFactory : public virtual IRef {
 public:
  [[nodiscard]] virtual Result<IReadByteCountMonitor> create(Filter* monitoredFilter) noexcept = 0;

  ABSTRACT_IREF(IReadByteCountMonitorFactory);
};

#endif  // GPUSDRPIPELINE_FILTERFACTORIES_H
