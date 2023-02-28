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

#include "AacFileWriter.h"

#include <iostream>

#include "GSErrors.h"
#include "GSLog.h"
#include "util/ScopeExit.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/buffer.h>
}

using namespace std;

#define SAFE_FF(ffmpegCmd__)                                                                                   \
  do {                                                                                                         \
    int ffmpegStatus__ = (ffmpegCmd__);                                                                        \
    if (ffmpegStatus__ < 0 && ffmpegStatus__ != AVERROR(EAGAIN) && ffmpegStatus__ != AVERROR_EOF) {            \
      char ffmpegErrorDescription__[128];                                                                      \
      av_make_error_string(ffmpegErrorDescription__, sizeof(ffmpegErrorDescription__), ffmpegStatus__);        \
                                                                                                               \
      gsloge("FFmpeg error [%d]: %s. At %s:%d", ffmpegStatus__, ffmpegErrorDescription__, __FILE__, __LINE__); \
      throw runtime_error("FFmpeg Error");                                                                     \
    }                                                                                                          \
  } while (false)

#define SAFE_FF_ONLY_LOG(ffmpegCmd__)                                                                        \
  do {                                                                                                       \
    int ffmpegStatus__ = (ffmpegCmd__);                                                                      \
    if (ffmpegStatus__ < 0 && ffmpegStatus__ != AVERROR(EAGAIN) && ffmpegStatus__ != AVERROR_EOF) {          \
      char ffmpegErrorDescription__[128];                                                                    \
      av_make_error_string(ffmpegErrorDescription__, sizeof(ffmpegErrorDescription__), ffmpegStatus__);      \
                                                                                                             \
      gsloge("FFmpeg error %d: %s. At %s:%d", ffmpegStatus__, ffmpegErrorDescription__, __FILE__, __LINE__); \
    }                                                                                                        \
  } while (false)

static Status ffmpegErrToStatus(int ffmpegStatus) noexcept {
  switch (ffmpegStatus) {
    case AVERROR(ENOMEM):
      return Status_OutOfMemory;

    case AVERROR(EINVAL):
      return Status_InvalidArgument;

    default:
      return ffmpegStatus >= 0 ? Status_Success : Status_RuntimeError;
  }
}

#define SAFE_FF_RET(ffmpegCmd__)                                                                             \
  do {                                                                                                       \
    int ffmpegStatus__ = (ffmpegCmd__);                                                                      \
    if (ffmpegStatus__ < 0 && ffmpegStatus__ != AVERROR(EAGAIN) && ffmpegStatus__ != AVERROR_EOF) {          \
      char ffmpegErrorDescription__[128];                                                                    \
      av_make_error_string(ffmpegErrorDescription__, sizeof(ffmpegErrorDescription__), ffmpegStatus__);      \
                                                                                                             \
      gsloge("FFmpeg error %d: %s. At %s:%d", ffmpegStatus__, ffmpegErrorDescription__, __FILE__, __LINE__); \
                                                                                                             \
      return ffmpegErrToStatus(ffmpegStatus__);                                                              \
    }                                                                                                        \
  } while (false)

static AVCodecContext* createCodecCtx(
    int32_t sampleRate,
    int32_t bitRate,
    const shared_ptr<AVFormatContext>& formatCtx) {
  const AVCodecID codecId = AV_CODEC_ID_AAC;
  const AVCodec* audioCodec = avcodec_find_encoder(codecId);
  if (audioCodec == nullptr) {
    GS_FAIL("Could not find the audio codec [" << avcodec_get_name(codecId) << "]");
  }

  AVCodecContext* audioCodecCtx = avcodec_alloc_context3(audioCodec);
  if (audioCodecCtx == nullptr) {
    GS_FAIL("Failed to allocate an AVCodecContext");
  }

  audioCodecCtx->bit_rate = bitRate;
  audioCodecCtx->sample_rate = sampleRate;
  audioCodecCtx->sample_fmt = AV_SAMPLE_FMT_FLT;
  audioCodecCtx->time_base = AVRational {1, sampleRate};
  av_channel_layout_default(&audioCodecCtx->ch_layout, 1);

  if (formatCtx->oformat->flags & AVFMT_GLOBALHEADER) {
    audioCodecCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

  int status = avcodec_open2(audioCodecCtx, audioCodec, nullptr);
  if (status < 0) {
    SAFE_FF_ONLY_LOG(status);
    avcodec_free_context(&audioCodecCtx);
    return nullptr;
  }

  return audioCodecCtx;
}

static AVFormatContext* createFormatCtx(const char* outputFileName) {
  AVFormatContext* formatCtxRaw = nullptr;
  SAFE_FF(avformat_alloc_output_context2(&formatCtxRaw, nullptr, nullptr, outputFileName));
  if (formatCtxRaw == nullptr) {
    throw bad_alloc();
  }

  return formatCtxRaw;
}

Status AacFileWriter::openFormatCtx() noexcept {
  if (mOpenedFormatCtx) {
    return Status_Success;
  }

  mOpenedFormatCtx = true;

  AVStream* stream = avformat_new_stream(mFormatCtx.get(), mAudioCodecCtx->codec);
  if (stream == nullptr) {
    return Status_OutOfMemory;
  }

  SAFE_FF_RET(avcodec_parameters_from_context(stream->codecpar, mAudioCodecCtx.get()));

  if ((mFormatCtx->oformat->flags & AVFMT_NOFILE) == 0) {
    SAFE_FF_RET(avio_open(&mFormatCtx->pb, mOutputFileName.c_str(), AVIO_FLAG_WRITE));
  }

  SAFE_FF_RET(avformat_write_header(mFormatCtx.get(), nullptr));

  return Status_Success;
}

static shared_ptr<AVFrame> createAvFrame() {
  return {av_frame_alloc(), [](AVFrame* frame) { av_frame_free(&frame); }};
}

static size_t getCodecInputBufferSize(const shared_ptr<AVCodecContext>& codecCtx) {
  if ((codecCtx->codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE) == 0) {
    GS_REQUIRE_OR_THROW_FMT(codecCtx->frame_size > 0, "Unexpected frame size [%d]", codecCtx->frame_size);

    return codecCtx->frame_size * sizeof(float) * codecCtx->ch_layout.nb_channels;
  }

  return 1024 * sizeof(float) * 2;
}

static shared_ptr<AVBufferPool> createBufferPool(const shared_ptr<AVCodecContext>& codecCtx) {
  const size_t bufferSize = getCodecInputBufferSize(codecCtx);
  AVBufferPool* bufferPoolRaw = av_buffer_pool_init(bufferSize, nullptr);
  if (bufferPoolRaw == nullptr) {
    GS_FAIL("Failed to create buffer pool");
  }

  return {bufferPoolRaw, [](AVBufferPool* pool) { av_buffer_pool_uninit(&pool); }};
}

static shared_ptr<AVPacket> createAvPacket() {
  AVPacket* avPacketRaw = av_packet_alloc();
  if (avPacketRaw == nullptr) {
    gsloge("Failed to create AVPcket: Out of memory");
    throw bad_alloc();
  }

  return {avPacketRaw, [](AVPacket* pkt) { av_packet_free(&pkt); }};
}

Result<Sink> AacFileWriter::create(
    const char* outputFileName,
    int32_t sampleRate,
    int32_t bitRate,
    int32_t cudaDevice,
    cudaStream_t cudaStream,
    IFactories* factories) noexcept {
  Ref<IAllocator> sysMemAllocator = factories->getSysMemAllocator();
  Ref<IBufferCopier> sysMemBufferCopier = factories->getSysMemCopier();
  Ref<IBufferSliceFactory> bufferSliceFactory = factories->getBufferSliceFactory();

  Ref<IRelocatableResizableBufferFactory> sysMemRelocatableBufferFactory;
  UNWRAP_OR_FWD_RESULT(
      sysMemRelocatableBufferFactory,
      factories->createRelocatableResizableBufferFactory(sysMemAllocator.get(), sysMemBufferCopier.get()));

  DO_OR_RET_ERR_RESULT(return makeRefResultNonNull<Sink>(new (nothrow) AacFileWriter(
      outputFileName,
      sampleRate,
      bitRate,
      cudaDevice,
      cudaStream,
      sysMemAllocator.get(),
      sysMemRelocatableBufferFactory.get(),
      bufferSliceFactory.get())));
}

AacFileWriter::AacFileWriter(
    const char* outputFileName,
    int32_t sampleRate,
    int32_t bitRate,
    int32_t cudaDevice,
    cudaStream_t cudaStream,
    IAllocator* sysMemAllocator,
    IRelocatableResizableBufferFactory* sysMemRelocatableBufferFactory,
    IBufferSliceFactory* bufferSliceFactory)
    : BaseSink(sysMemRelocatableBufferFactory, bufferSliceFactory, 1),
      mOutputFileName(outputFileName),
      mAllocator(sysMemAllocator),
      mFormatCtx(shared_ptr<AVFormatContext>(
          createFormatCtx(outputFileName),
          [this](AVFormatContext* ctx) {
            SAFE_FF_ONLY_LOG(closeFormatContext());
            avformat_free_context(ctx);
          })),
      mAudioCodecCtx(shared_ptr<AVCodecContext>(
          createCodecCtx(sampleRate, bitRate, mFormatCtx),
          [this](AVCodecContext* ctx) {
            SAFE_FF_ONLY_LOG(closeCodecContext());
            avcodec_free_context(&ctx);
          })),
      mInputFrame(createAvFrame()),
      mOutputPacket(createAvPacket()),
      mAudioCodecInputBufferSize(getCodecInputBufferSize(mAudioCodecCtx)),
      mBufferPool(createBufferPool(mAudioCodecCtx)),
      mOpenedFormatCtx(false),
      mClosedFormatCtx(false),
      mClosedCodecCtx(false),
      mSentSampleCount(0),
      mCudaWaiter(cudaDevice, cudaStream) {
  gslogd(
      "Created AAC File writer. Output file [%s] sample rate [%d] bit rate [%d]",
      outputFileName,
      sampleRate,
      bitRate);
}

AacFileWriter::~AacFileWriter() {
  auto inputBuffer = getPortInputBuffer(0).value;
  const size_t availableByteCount = inputBuffer == nullptr ? 0 : inputBuffer->range()->used();

  if (availableByteCount > 0) {
    try {
      SAFE_FF_ONLY_LOG(encodeAndMuxAvailable(true /* flush */, 0));
    } catch (exception& e) {
      gslogw("Failed to flush AacFileWriter output: %s", e.what());
    } catch (...) {
      gslogw("Unknown error flushing AacFileWriter.");
    }
  }
}

Result<IBuffer> AacFileWriter::requestBuffer(size_t port, size_t numBytes) noexcept {
  return BaseSink::requestBuffer(port, numBytes);
}

Status AacFileWriter::commitBuffer(size_t port, size_t numBytes) noexcept {
  Status status = BaseSink::commitBuffer(port, numBytes);
  if (status != Status_Success) {
    return status;
  }

  FWD_IF_ERR(mCudaWaiter.recordNextAndWaitPrevious());

  const size_t numBytesInFlightToExclude = numBytes;

  SAFE_FF_RET(encodeAndMuxAvailable(false /* flush */, numBytesInFlightToExclude));

  return Status_Success;
}

Status AacFileWriter::encodeAndMuxAvailable(bool flush, size_t excludeByteCountInFlight) noexcept {
  ConstRef<IBuffer> inputBuffer = unwrap(getPortInputBuffer(0));

  while (inputBuffer->range()->used() - excludeByteCountInFlight >= mAudioCodecInputBufferSize) {
    SAFE_FF_RET(sendOneAudioBuffer(false /* flush */, excludeByteCountInFlight));
    SAFE_FF_RET(writeAvailablePackets());
  }

  if (flush && inputBuffer->range()->used() > excludeByteCountInFlight) {
    RET_IF_ERR(
        mCudaWaiter.recordNextAndWaitPrevious(),
        Status_InvalidState);  // Case happens in destructor, waits on the previous commitBuffer() data.
    SAFE_FF_RET(sendOneAudioBuffer(true /* flush */, excludeByteCountInFlight));
    SAFE_FF_RET(writeAvailablePackets());
  }

  FWD_IF_ERR(
      consumeInputBytesAndMoveUsedToStart(0, 0));  // The input buffer offset is increased by sendOneAudioBuffer().

  return Status_Success;
}

Status AacFileWriter::sendOneAudioBuffer(bool flush, size_t excludeByteCountInFlight) noexcept {
  ConstRef<IBuffer> inputBuffer = unwrap(getPortInputBuffer(0));

  const size_t availableInputByteCount = inputBuffer->range()->used() - excludeByteCountInFlight;
  const bool haveFullBuffer = availableInputByteCount >= mAudioCodecInputBufferSize;
  const bool sendEndOfStreamSignal = !haveFullBuffer && flush;

  if (!haveFullBuffer && !sendEndOfStreamSignal) {
    return Status_Success;
  }

  mInputFrame->buf[0] = av_buffer_pool_get(mBufferPool.get());
  if (mInputFrame->buf[0] == nullptr) {
    return Status_OutOfMemory;
  }

  ScopeExit frameUnreffer([frame = mInputFrame]() { av_frame_unref(frame.get()); });

  mInputFrame->data[0] = mInputFrame->buf[0]->data;

  const size_t useNumBytes = min(mAudioCodecInputBufferSize, availableInputByteCount) / sizeof(float) * sizeof(float);
  const size_t channelCount = mAudioCodecCtx->ch_layout.nb_channels;
  const size_t sampleCount = useNumBytes / sizeof(float) / channelCount;

  void* targetData = mInputFrame->data[0];
  const void* sourceData = inputBuffer->readPtr();
  memcpy(targetData, sourceData, useNumBytes);

  mInputFrame->pts = mSentSampleCount;
  mInputFrame->duration = static_cast<int64_t>(sampleCount);
  mInputFrame->nb_samples = static_cast<int>(sampleCount);

  SAFE_FF_RET(av_channel_layout_copy(&mInputFrame->ch_layout, &mAudioCodecCtx->ch_layout));

  mSentSampleCount += static_cast<int64_t>(sampleCount);

  SAFE_FF_RET(avcodec_send_frame(mAudioCodecCtx.get(), mInputFrame.get()));

  if (sendEndOfStreamSignal) {
    SAFE_FF_RET(avcodec_send_frame(mAudioCodecCtx.get(), nullptr));
  }

  FWD_IF_ERR(inputBuffer->range()->increaseOffset(useNumBytes));

  return Status_Success;
}

Status AacFileWriter::writeAvailablePackets() noexcept {
  int status = 0;

  do {
    status = avcodec_receive_packet(mAudioCodecCtx.get(), mOutputPacket.get());

    if (status == AVERROR(EAGAIN)) {
      return Status_Success;
    } else if (status == AVERROR_EOF) {
      WARN_IF_ERR(closeCodecContext());
      WARN_IF_ERR(closeFormatContext());

      return Status_Success;
    }
    SAFE_FF_RET(status);

    ScopeExit packetUnreffer([this]() { av_packet_unref(mOutputPacket.get()); });

    if (!mOpenedFormatCtx) {
      FWD_IF_ERR(openFormatCtx());
    }

    rescalePacketFromCodecToContainerTime(mOutputPacket);

    mOutputPacket->stream_index = 0;
    SAFE_FF_RET(av_interleaved_write_frame(mFormatCtx.get(), mOutputPacket.get()));
  } while (status >= 0);

  return Status_Success;
}

Status AacFileWriter::closeCodecContext() noexcept {
  if (mClosedCodecCtx) {
    return Status_Success;
  }

  mClosedCodecCtx = true;
  return ffmpegErrToStatus(avcodec_close(mAudioCodecCtx.get()));
}

Status AacFileWriter::closeFormatContext() noexcept {
  if (!mOpenedFormatCtx || mClosedFormatCtx) {
    return Status_Success;
  }

  mClosedFormatCtx = true;
  int writeTrailerStatus = av_write_trailer(mFormatCtx.get());
  int closeAvioStatus = 0;

  if ((mFormatCtx->oformat->flags & AVFMT_NOFILE) == 0) {
    closeAvioStatus = avio_closep(&mFormatCtx->pb);
  }

  SAFE_FF_RET(writeTrailerStatus);
  SAFE_FF_RET(closeAvioStatus);

  return Status_Success;
}

int64_t AacFileWriter::rescaleToContainerTime(int64_t codecTime) const noexcept {
  return av_rescale_q(codecTime, mAudioCodecCtx->time_base, mFormatCtx->streams[0]->time_base);
}

void AacFileWriter::rescalePacketFromCodecToContainerTime(const std::shared_ptr<AVPacket>& pkt) const noexcept {
  pkt->time_base = mFormatCtx->streams[0]->time_base;
  pkt->pts = rescaleToContainerTime(mOutputPacket->pts);

  if (pkt->dts != AV_NOPTS_VALUE) {
    pkt->dts = rescaleToContainerTime(mOutputPacket->dts);
  }

  if (pkt->duration > 0) {
    pkt->duration = rescaleToContainerTime(mOutputPacket->duration);
  }
}

size_t AacFileWriter::preferredInputBufferSize(size_t port) noexcept {
  GS_REQUIRE_OR_ABORT(port == 0, "Port is out of range");
  return mAudioCodecInputBufferSize;
}
