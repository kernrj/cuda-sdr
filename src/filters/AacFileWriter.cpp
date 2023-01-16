//
// Created by Rick Kern on 1/5/23.
//

#include "AacFileWriter.h"

#include <iostream>

#include "GSErrors.h"
#include "util/ScopeExit.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/buffer.h>
}

using namespace std;

#define SAFE_FF(ffmpegCmd__)                                                                                  \
  do {                                                                                                        \
    int ffmpegStatus__ = (ffmpegCmd__);                                                                       \
    if (ffmpegStatus__ < 0 && ffmpegStatus__ != AVERROR(EAGAIN) && ffmpegStatus__ != AVERROR_EOF) {           \
      char ffmpegErrorDescription__[128];                                                                     \
      av_make_error_string(ffmpegErrorDescription__, sizeof(ffmpegErrorDescription__), ffmpegStatus__);       \
                                                                                                              \
      THROW(                                                                                                  \
          "FFmpeg error " << ffmpegStatus__ << ": " << ffmpegErrorDescription__ << ". At " << __FILE__ << ':' \
                          << __LINE__);                                                                       \
    }                                                                                                         \
  } while (false)

#define SAFE_FF_ONLY_LOG(ffmpegCmd__)                                                                   \
  do {                                                                                                  \
    int ffmpegStatus__ = (ffmpegCmd__);                                                                 \
    if (ffmpegStatus__ < 0 && ffmpegStatus__ != AVERROR(EAGAIN) && ffmpegStatus__ != AVERROR_EOF) {     \
      char ffmpegErrorDescription__[128];                                                               \
      av_make_error_string(ffmpegErrorDescription__, sizeof(ffmpegErrorDescription__), ffmpegStatus__); \
                                                                                                        \
      fprintf(                                                                                          \
          stderr,                                                                                       \
          "FFmpeg error %d: %s. At %s:%d\n",                                                            \
          ffmpegStatus__,                                                                               \
          ffmpegErrorDescription__,                                                                     \
          __FILE__,                                                                                     \
          __LINE__);                                                                                    \
    }                                                                                                   \
  } while (false)

#define SAFE_FF_RET(ffmpegCmd__)                                                                                  \
  do {                                                                                                            \
    int ffmpegStatus__ = (ffmpegCmd__);                                                                           \
    if (ffmpegStatus__ < 0 && ffmpegStatus__ != AVERROR(EAGAIN) && ffmpegStatus__ != AVERROR_EOF) {               \
      char ffmpegErrorDescription__[128];                                                                         \
      av_make_error_string(ffmpegErrorDescription__, sizeof(ffmpegErrorDescription__), ffmpegStatus__);           \
                                                                                                                  \
      cerr << "FFmpeg error " << ffmpegStatus__ << ": " << ffmpegErrorDescription__ << ". At " << __FILE__ << ':' \
           << __LINE__ << endl;                                                                                   \
                                                                                                                  \
      return ffmpegStatus__;                                                                                      \
    }                                                                                                             \
  } while (false)

static AVCodecContext* createCodecCtx(
    int32_t sampleRate,
    int32_t bitRate,
    const shared_ptr<AVFormatContext>& formatCtx) {
  const AVCodecID codecId = AV_CODEC_ID_AAC;
  const AVCodec* audioCodec = avcodec_find_encoder(codecId);
  if (audioCodec == nullptr) {
    THROW("Could not find the audio codec [" << avcodec_get_name(codecId) << "]");
  }

  AVCodecContext* audioCodecCtx = avcodec_alloc_context3(audioCodec);
  if (audioCodecCtx == nullptr) {
    throw bad_alloc();
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

static AVFormatContext* createFormatCtx(const string& outputFileName) {
  AVFormatContext* formatCtxRaw = nullptr;
  SAFE_FF(avformat_alloc_output_context2(&formatCtxRaw, nullptr, nullptr, outputFileName.c_str()));

  if (formatCtxRaw == nullptr) {
    THROW("Could not allocate output format context");
  }

  return formatCtxRaw;
}

int AacFileWriter::openFormatCtx() {
  if (mOpenedFormatCtx) {
    return 0;
  }

  mOpenedFormatCtx = true;

  AVStream* stream = avformat_new_stream(mFormatCtx.get(), mAudioCodecCtx->codec);
  if (stream == nullptr) {
    return AVERROR(ENOMEM);
  }

  SAFE_FF_RET(avcodec_parameters_from_context(stream->codecpar, mAudioCodecCtx.get()));

  if ((mFormatCtx->oformat->flags & AVFMT_NOFILE) == 0) {
    SAFE_FF_RET(avio_open(&mFormatCtx->pb, mOutputFileName.c_str(), AVIO_FLAG_WRITE));
  }

  SAFE_FF_RET(avformat_write_header(mFormatCtx.get(), nullptr));

  return 0;
}

static shared_ptr<AVFrame> createAvFrame() {
  return {av_frame_alloc(), [](AVFrame* frame) { av_frame_free(&frame); }};
}

static size_t getCodecInputBufferSize(const shared_ptr<AVCodecContext>& codecCtx) {
  if ((codecCtx->codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE) == 0) {
    if (codecCtx->frame_size <= 0) {
      THROW("Unexpected frame size [" << codecCtx->frame_size << "]");
    }

    return codecCtx->frame_size * sizeof(float) * codecCtx->ch_layout.nb_channels;
  }

  return 1024 * sizeof(float) * 2;
}

static shared_ptr<AVBufferPool> createBufferPool(const shared_ptr<AVCodecContext>& codecCtx) {
  const size_t bufferSize = getCodecInputBufferSize(codecCtx);
  AVBufferPool* bufferPoolRaw = av_buffer_pool_init(bufferSize, nullptr);
  if (bufferPoolRaw == nullptr) {
    THROW("Failed to create buffer pool");
  }

  return {bufferPoolRaw, [](AVBufferPool* pool) { av_buffer_pool_uninit(&pool); }};
}

static shared_ptr<AVPacket> createAvPacket() {
  AVPacket* avPacketRaw = av_packet_alloc();
  if (avPacketRaw == nullptr) {
    THROW("Failed to create output packet");
  }

  return {avPacketRaw, [](AVPacket* pkt) { av_packet_free(&pkt); }};
}

AacFileWriter::AacFileWriter(const string& outputFileName, int32_t sampleRate, int32_t bitRate, IFactories* factories)
    : BaseSink(factories->getRelocatableSysMemBufferFactory(), factories->getBufferSliceFactory(), 1),
      mOutputFileName(outputFileName),
      mAllocator(factories->getSysMemAllocator()),
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
      mSentSampleCount(0) {}

AacFileWriter::~AacFileWriter() {
  auto inputBuffer = getPortInputBuffer(0);

  if (inputBuffer != nullptr && inputBuffer->range()->used() > 0) {
    size_t usedByteCount = 0;
    SAFE_FF_ONLY_LOG(encodeAndMuxAvailable(true /* flush */));
  }
}

std::shared_ptr<IBuffer> AacFileWriter::requestBuffer(size_t port, size_t numBytes) {
  return BaseSink::requestBuffer(port, numBytes);
}

void AacFileWriter::commitBuffer(size_t port, size_t numBytes) {
  BaseSink::commitBuffer(port, numBytes);

  /*
  static float rate = 2.0f * M_PIf * 440 / mAudioCodecCtx->sample_rate;
  auto inputBuffer = getPortInputBuffer(0);
  const size_t floatCount = numBytes / sizeof(float);

  float* writePtr = (float*)(inputBuffer->writePtr() - numBytes);
  for (size_t i = 0; i < floatCount; i++) {
    t += rate;
    writePtr[i] = cos(t);
    printf("%zu (%zu)=%f ", i, i + inputBuffer->range()->offset(), writePtr[i]);
  }
   */

  SAFE_FF(encodeAndMuxAvailable(false /* flush */));
}

int AacFileWriter::encodeAndMuxAvailable(bool flush) {
  auto inputBuffer = getPortInputBuffer(0);

  while (inputBuffer->range()->used() >= mAudioCodecInputBufferSize) {
    SAFE_FF_RET(sendOneAudioBuffer(false /* flush */));
    SAFE_FF_RET(writeAvailablePackets());
  }

  if (flush && inputBuffer->range()->used() > 0) {
    SAFE_FF_RET(sendOneAudioBuffer(true /* flush */));
    SAFE_FF_RET(writeAvailablePackets());
  }

  const float* samples = inputBuffer->readPtr<float>();
  consumeInputBytesAndMoveUsedToStart(0, 0);  // The input buffer offset is increased by sendOneAudioBuffer().

  samples = inputBuffer->readPtr<float>();

  return 0;
}

int AacFileWriter::sendOneAudioBuffer(bool flush) {
  const shared_ptr<IBuffer> inputBuffer = getPortInputBuffer(0);

  const size_t usedByteCount = inputBuffer->range()->used();
  const bool haveFullBuffer = usedByteCount >= mAudioCodecInputBufferSize;
  const bool haveAnyData = usedByteCount > 0;
  const bool sendEndOfStreamSignal = !haveFullBuffer && flush;

  if (!haveFullBuffer && !sendEndOfStreamSignal) {
    return 0;
  }

  mInputFrame->buf[0] = av_buffer_pool_get(mBufferPool.get());
  if (mInputFrame->buf[0] == nullptr) {
    return AVERROR(ENOMEM);
  }

  ScopeExit frameUnreffer([frame = mInputFrame]() { av_frame_unref(frame.get()); });

  mInputFrame->data[0] = mInputFrame->buf[0]->data;

  const size_t useNumBytes = min(mAudioCodecInputBufferSize, inputBuffer->range()->used());
  const size_t channelCount = mAudioCodecCtx->ch_layout.nb_channels;
  const size_t sampleCount = useNumBytes / sizeof(float) / channelCount;

  float* targetData = (float*)mInputFrame->data[0];
  const float* sourceData = inputBuffer->readPtr<float>();
  memcpy(targetData, sourceData, useNumBytes);

  mInputFrame->pts = mSentSampleCount;
  mInputFrame->duration = sampleCount;
  mInputFrame->nb_samples = sampleCount;

  SAFE_FF_RET(av_channel_layout_copy(&mInputFrame->ch_layout, &mAudioCodecCtx->ch_layout));

  mSentSampleCount += sampleCount;

  SAFE_FF_RET(avcodec_send_frame(mAudioCodecCtx.get(), mInputFrame.get()));

  if (sendEndOfStreamSignal) {
    SAFE_FF_RET(avcodec_send_frame(mAudioCodecCtx.get(), nullptr));
  }

  inputBuffer->range()->increaseOffset(useNumBytes);

  return 0;
}

int AacFileWriter::writeAvailablePackets() {
  int status = 0;

  do {
    status = avcodec_receive_packet(mAudioCodecCtx.get(), mOutputPacket.get());

    if (status == AVERROR(EAGAIN)) {
      return 0;
    } else if (status == AVERROR_EOF) {
      int closeCodecStatus = closeCodecContext();
      int closeFormatStatus = closeFormatContext();

      SAFE_FF_RET(closeCodecStatus);
      SAFE_FF_RET(closeFormatStatus);

      return 0;
    }
    SAFE_FF_RET(status);

    ScopeExit packetUnreffer([this]() { av_packet_unref(mOutputPacket.get()); });

    if (!mOpenedFormatCtx) {
      SAFE_FF_RET(openFormatCtx());
    }

    rescalePacketFromCodecToContainerTime(mOutputPacket);

    mOutputPacket->stream_index = 0;
    SAFE_FF_RET(av_interleaved_write_frame(mFormatCtx.get(), mOutputPacket.get()));
  } while (status >= 0);

  return 0;
}

int AacFileWriter::closeCodecContext() {
  if (mClosedCodecCtx) {
    return 0;
  }

  mClosedCodecCtx = true;
  return avcodec_close(mAudioCodecCtx.get());
}

int AacFileWriter::closeFormatContext() {
  if (!mOpenedFormatCtx || mClosedFormatCtx) {
    return 0;
  }

  mClosedFormatCtx = true;
  int writeTrailerStatus = av_write_trailer(mFormatCtx.get());
  int closeAvioStatus = 0;

  if ((mFormatCtx->oformat->flags & AVFMT_NOFILE) == 0) {
    closeAvioStatus = avio_closep(&mFormatCtx->pb);
  }

  SAFE_FF_RET(writeTrailerStatus);
  SAFE_FF_RET(closeAvioStatus);

  return 0;
}

int64_t AacFileWriter::rescaleToContainerTime(int64_t codecTime) const {
  return av_rescale_q(codecTime, mAudioCodecCtx->time_base, mFormatCtx->streams[0]->time_base);
}

void AacFileWriter::rescalePacketFromCodecToContainerTime(const std::shared_ptr<AVPacket>& pkt) const {
  pkt->time_base = mFormatCtx->streams[0]->time_base;
  pkt->pts = rescaleToContainerTime(mOutputPacket->pts);

  if (pkt->dts != AV_NOPTS_VALUE) {
    pkt->dts = rescaleToContainerTime(mOutputPacket->dts);
  }

  if (pkt->duration > 0) {
    pkt->duration = rescaleToContainerTime(mOutputPacket->duration);
  }
}
