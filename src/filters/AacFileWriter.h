//
// Created by Rick Kern on 1/5/23.
//

#ifndef GPUSDR_AACFILEWRITER_H
#define GPUSDR_AACFILEWRITER_H

#include "Factories.h"
#include "buffers/IAllocator.h"
#include "filters/BaseSink.h"

struct AVBufferPool;
struct AVCodecContext;
struct AVFormatContext;
struct AVFrame;
struct AVPacket;

class AacFileWriter : public BaseSink {
 public:
  AacFileWriter(const std::string& outputFileName, int32_t sampleRate, int32_t bitRate, IFactories* factories);
  ~AacFileWriter() override;

  [[nodiscard]] std::shared_ptr<IBuffer> requestBuffer(size_t port, size_t numBytes) override;
  void commitBuffer(size_t port, size_t numBytes) override;

 private:
  const std::string mOutputFileName;
  const std::shared_ptr<IAllocator> mAllocator;
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

 private:
  [[nodiscard]] int encodeAndMuxAvailable(bool flush);
  [[nodiscard]] int sendOneAudioBuffer(bool flush);
  [[nodiscard]] int writeAvailablePackets();
  [[nodiscard]] int closeCodecContext();
  [[nodiscard]] int openFormatCtx();
  [[nodiscard]] int closeFormatContext();
  [[nodiscard]] int64_t rescaleToContainerTime(int64_t codecTime) const;
  void rescalePacketFromCodecToContainerTime(const std::shared_ptr<AVPacket>& pkt) const;
};

#endif  // GPUSDR_AACFILEWRITER_H
