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

#ifndef GPUSDRPIPELINE_RFTOPCMAUDIO_H
#define GPUSDRPIPELINE_RFTOPCMAUDIO_H

#include <gpusdrpipeline/Factories.h>
#include <gpusdrpipeline/Modulation.h>
#include <remez/remez.h>

class RfToPcmAudio final : public Filter {
 public:
  static Result<Filter> create(
      float rfSampleRate,
      Modulation modulation,
      size_t rfLowPassDecim,
      size_t audioLowPassDecim,  // audio sample rate is rfSampleRate / rfLowPassDecimation / audioLowPassDecim
      float centerFrequency,
      float channelFrequency,
      float channelWidth,
      float fskDevationIfFm,
      float rfLowPassDbAttenuation,
      float audioLowPassDbAttenuation,
      int32_t cudaDevice,
      cudaStream_t cudaStream,
      IFactories* factories) noexcept;

  [[nodiscard]] Result<IBuffer> requestBuffer(size_t port, size_t byteCount) noexcept final;
  [[nodiscard]] Status commitBuffer(size_t port, size_t byteCount) noexcept final;
  [[nodiscard]] size_t getOutputDataSize(size_t port) noexcept final;
  [[nodiscard]] size_t getOutputSizeAlignment(size_t port) noexcept final;
  Status readOutput(IBuffer** portOutputBuffers, size_t numPorts) noexcept final;
  [[nodiscard]] size_t preferredInputBufferSize(size_t port) noexcept final;

 private:
  ConstRef<IFilterDriver> mDriver;
  ConstRef<Source> mCosineSource;
  ConstRef<Filter> mMultiplyRfSourceByCosine;
  ConstRef<Filter> mRfLowPassFilter;
  ConstRef<Filter> mQuadDemod;
  ConstRef<Filter> mAudioLowPassFilter;

 private:
  RfToPcmAudio(
      IFilterDriver* driver,
      Source* cosineSource,
      Filter* multiply,
      Filter* rfLowPassFilter,
      Filter* quadDemodFilter,
      Filter* audioLowPassFilter) noexcept;

  REF_COUNTED(RfToPcmAudio);
};

#endif  // GPUSDRPIPELINE_RFTOPCMAUDIO_H
