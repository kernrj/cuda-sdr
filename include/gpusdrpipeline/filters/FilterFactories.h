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

#include <gpusdrpipeline/Modulation.h>
#include <gpusdrpipeline/SampleType.h>
#include <gpusdrpipeline/commandqueue/ICudaCommandQueue.h>
#include <gpusdrpipeline/filters/Filter.h>
#include <gpusdrpipeline/filters/IHackrfSource.h>
#include <gpusdrpipeline/filters/IPortRemappingSink.h>
#include <gpusdrpipeline/filters/IReadByteCountMonitor.h>

#include "IPortRemappingSource.h"

class INodeFactory : public virtual IRef {
 public:
  virtual Result<Node> create(const char* jsonParameters) noexcept = 0;
  ABSTRACT_IREF(INodeFactory);
};

GS_EXPORT [[nodiscard]] Result<Node> createNode(const char* name, const char* jsonParameters) noexcept;
GS_EXPORT [[nodiscard]] Result<Filter> createFilter(const char* name, const char* jsonParameters) noexcept;
GS_EXPORT [[nodiscard]] Result<Source> createSource(const char* name, const char* jsonParameters) noexcept;
GS_EXPORT [[nodiscard]] Result<Sink> createSink(const char* name, const char* jsonParameters) noexcept;

GS_EXPORT [[nodiscard]] bool hasNodeFactory(const char* name) noexcept;
GS_EXPORT [[nodiscard]] Status registerNodeFactory(const char* name, INodeFactory* filterFactory) noexcept;
GS_EXPORT [[nodiscard]] Status registerDefaultNodeFactories() noexcept;

class ICudaMemcpyFilterFactory : public INodeFactory {
 public:
  [[nodiscard]] virtual Result<Filter> createCudaMemcpy(
      cudaMemcpyKind memcpyKind,
      ICudaCommandQueue* commandQueue) noexcept = 0;

  ABSTRACT_IREF(ICudaMemcpyFilterFactory);
};

class IAacFileWriterFactory : public INodeFactory {
 public:
  [[nodiscard]] virtual Result<Sink> createAacFileWriter(
      const char* outputFileName,
      int32_t sampleRate,
      int32_t bitRate,
      ICudaCommandQueue* commandQueue) noexcept = 0;

  ABSTRACT_IREF(IAacFileWriterFactory);
};

class IAddConstFactory : public INodeFactory {
 public:
  [[nodiscard]] virtual Result<Filter> createAddConst(
      float addValueToAmplitude,
      ICudaCommandQueue* commandQueue) noexcept = 0;

  ABSTRACT_IREF(IAddConstFactory);
};

class IAddConstToVectorLengthFactory : public INodeFactory {
 public:
  [[nodiscard]] virtual Result<Filter> createAddConstToVectorLength(
      float addValueToMagnitude,
      ICudaCommandQueue* commandQueue) noexcept = 0;

  ABSTRACT_IREF(IAddConstToVectorLengthFactory);
};

class ICosineSourceFactory : public INodeFactory {
 public:
  [[nodiscard]] virtual Result<Source> createCosineSource(
      SampleType sampleType,
      float sampleRate,
      float frequency,
      ICudaCommandQueue* commandQueue) noexcept = 0;

  ABSTRACT_IREF(ICosineSourceFactory);
};

class IFileReaderFactory : public INodeFactory {
 public:
  [[nodiscard]] virtual Result<Source> createFileReader(const char* fileName) noexcept = 0;

  ABSTRACT_IREF(IFileReaderFactory);
};

class IFirFactory : public INodeFactory {
 public:
  [[nodiscard]] virtual Result<Filter> createFir(
      SampleType tapType,
      SampleType elementType,
      size_t decimation,
      const float* taps,
      size_t tapCount,
      ICudaCommandQueue* commandQueue) noexcept = 0;

  ABSTRACT_IREF(IFirFactory);
};

class IHackrfSourceFactory : public INodeFactory {
 public:
  [[nodiscard]] virtual Result<IHackrfSource> createHackrfSource(
      int32_t deviceIndex,
      uint64_t centerFrequency,
      double sampleRate,
      size_t maxBufferCountBeforeDropping) noexcept = 0;

  ABSTRACT_IREF(IHackrfSourceFactory);
};

class ICudaFilterFactory : public INodeFactory {
 public:
  [[nodiscard]] virtual Result<Filter> createFilter(ICudaCommandQueue* commandQueue) noexcept = 0;

  ABSTRACT_IREF(ICudaFilterFactory);
};

class IPortRemappingSinkFactory : public virtual IRef {
 public:
  [[nodiscard]] virtual Result<IPortRemappingSink> create() noexcept = 0;

  ABSTRACT_IREF(IPortRemappingSinkFactory);
};

class IPortRemappingSourceFactory : public virtual IRef {
 public:
  [[nodiscard]] virtual Result<IPortRemappingSource> create() noexcept = 0;

  ABSTRACT_IREF(IPortRemappingSourceFactory);
};

class IQuadDemodFactory : public INodeFactory {
 public:
  /**
   * @param fskDeviation Only used for FM
   */
  [[nodiscard]] virtual Result<Filter> createQuadDemod(
      Modulation modulation,
      float rfSampleRate,
      float fskDeviation,
      ICudaCommandQueue* commandQueue) noexcept = 0;
  ABSTRACT_IREF(IQuadDemodFactory);
};

class IRfToPcmAudioFactory : public INodeFactory {
 public:
  [[nodiscard]] virtual Result<Filter> createRfToPcm(
      float rfSampleRate,
      Modulation modulation,
      size_t rfLowPassDecim,
      size_t audioLowPassDecim,
      float centerFrequency,
      float channelFrequency,
      float channelWidth,
      float fskDeviationIfFm,
      float rfLowPassDbAttenuation,
      float audioLowPassDbAttenuation,
      const char* commandQueueId) noexcept = 0;

  ABSTRACT_IREF(IRfToPcmAudioFactory);
};

class IReadByteCountMonitorFactory : public virtual IRef {
 public:
  [[nodiscard]] virtual Result<IReadByteCountMonitor> create(Filter* monitoredFilter) noexcept = 0;

  ABSTRACT_IREF(IReadByteCountMonitorFactory);
};

#endif  // GPUSDRPIPELINE_FILTERFACTORIES_H
