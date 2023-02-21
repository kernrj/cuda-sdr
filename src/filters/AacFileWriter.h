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

#ifndef GPUSDR_AACFILEWRITER_H
#define GPUSDR_AACFILEWRITER_H

#include <memory>

#include "Factories.h"
#include "Waiter.h"
#include "buffers/IAllocator.h"
#include "filters/BaseSink.h"

struct AVBufferPool;
struct AVCodecContext;
struct AVFormatContext;
struct AVFrame;
struct AVPacket;

class AacFileWriter final : public BaseSink {
 public:
  static Result<Sink> create(
      const char* outputFileName,
      int32_t sampleRate,
      int32_t bitRate,
      int32_t cudaDevice,
      cudaStream_t cudaStream,
      IFactories* factories) noexcept;

  [[nodiscard]] Result<IBuffer> requestBuffer(size_t port, size_t numBytes) noexcept final;
  [[nodiscard]] Status commitBuffer(size_t port, size_t numBytes) noexcept final;

 private:
  const std::string mOutputFileName;
  ConstRef<IAllocator> mAllocator;
  const std::shared_ptr<AVFormatContext> mFormatCtx;
  const std::shared_ptr<AVCodecContext> mAudioCodecCtx;
  const std::shared_ptr<AVFrame> mInputFrame;
  const std::shared_ptr<AVPacket> mOutputPacket;
  const size_t mAudioCodecInputBufferSize;
  const std::shared_ptr<AVBufferPool> mBufferPool;
  bool mOpenedFormatCtx;
  bool mClosedFormatCtx;
  bool mClosedCodecCtx;
  int64_t mSentSampleCount;
  Waiter mCudaWaiter;

 private:
  [[nodiscard]] Status encodeAndMuxAvailable(bool flush, size_t excludeByteCountInFlight) noexcept;
  [[nodiscard]] Status sendOneAudioBuffer(bool flush, size_t excludeByteCountInFlight) noexcept;
  [[nodiscard]] Status writeAvailablePackets() noexcept;
  [[nodiscard]] Status closeCodecContext() noexcept;
  [[nodiscard]] Status openFormatCtx() noexcept;
  [[nodiscard]] Status closeFormatContext() noexcept;
  [[nodiscard]] int64_t rescaleToContainerTime(int64_t codecTime) const noexcept;
  void rescalePacketFromCodecToContainerTime(const std::shared_ptr<AVPacket>& pkt) const noexcept;

  AacFileWriter(
      const char* outputFileName,
      int32_t sampleRate,
      int32_t bitRate,
      int32_t cudaDevice,
      cudaStream_t cudaStream,
      IAllocator* sysMemAllocator,
      IRelocatableResizableBufferFactory* sysMemRelocatableBufferFactory,
      IBufferSliceFactory* bufferSliceFactory);

  ~AacFileWriter() final;

  REF_COUNTED_NO_DESTRUCTOR(AacFileWriter);
};

#endif  // GPUSDR_AACFILEWRITER_H
