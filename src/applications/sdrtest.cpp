/*
 * Copyright 2022 Rick Kern <kernrj@gmail.com>
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

#include <gnuradio/blocks/file_sink.h>
#include <gnuradio/blocks/file_source.h>
#include <gnuradio/blocks/wavfile_sink.h>
#include <gnuradio/filter/firdes.h>
#include <gnuradio/top_block.h>
#include <osmosdr/source.h>

#include <atomic>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <functional>
#include <mutex>
#include <stack>
#include <thread>

#include "../prototype/fm_pipeline.h"
#include "fm.h"

using namespace std;

static stack<function<void()>> runAtExit;
static atomic_bool wasInterrupted(false);
static condition_variable wasInterruptedCv;
static mutex shutdownMutex;

static void cleanupThings() {
  while (!runAtExit.empty()) {
    const auto toRun = std::move(runAtExit.top());
    runAtExit.pop();

    toRun();
  }
}

class Thread2 {
 public:
  explicit Thread2(function<void()>&& callable)
      : mThreadExited(false), mThread([this, threadFnc = std::move(callable)]() {
          threadFnc();

          {
            lock_guard<mutex> lock(mThreadExitedMutex);
            mThreadExited.store(true);
          }
          mThreadExitedCv.notify_all();
        }) {}

  template <typename Rep, typename Period>
  void joinWithTimeout(chrono::duration<Rep, Period> timeout) {
    unique_lock<mutex> lock(mThreadExitedMutex);
    while (!mThreadExited.load()) {
      cv_status status = mThreadExitedCv.wait_for(lock, timeout);

      if (!mThreadExited.load() && status == cv_status::timeout) {
        throw runtime_error("timed out waiting for thread to exit");
      }
    }

    mThread.join();
  }

  std::thread& getThread() { return mThread; }

 private:
  mutex mThreadExitedMutex;
  condition_variable mThreadExitedCv;
  atomic_bool mThreadExited;
  std::thread mThread;
};

static void waitForStop(const gr::top_block_sptr& topBlock) {
  Thread2 blockWaitThread([topBlock]() {
    topBlock->wait();

    printf("Graph ended\n");
    wasInterrupted.store(true);
    wasInterruptedCv.notify_all();
  });

  unique_lock<mutex> lock(shutdownMutex);
  wasInterruptedCv.wait(lock, []() { return wasInterrupted.load(); });

  blockWaitThread.joinWithTimeout(chrono::seconds(10));
}

static void exitSigHandler(int signum) {
  printf("Caught signal %d, cleaning up.\n", signum);
  cleanupThings();

  if (signum == SIGINT || signum == SIGTERM) {
    wasInterrupted.store(true);
    wasInterruptedCv.notify_all();
  } else {
    abort();
  }
}

int nbfm();
int fmRadio(int argc, char** argv);
int raw();
int nbfmFromRaw();

int main(int argc, char** argv) {
  atexit(cleanupThings);
  signal(SIGSEGV, &exitSigHandler);
  signal(SIGINT, &exitSigHandler);
  signal(SIGTERM, &exitSigHandler);

  return nbfmFromRaw();
  // return raw();
  // return nbfm();
  // return fmRadio(argc, argv);
}

int raw() {
  auto topBlock = gr::make_top_block("Top Block");
  runAtExit.push([topBlock]() { topBlock->stop(); });

  const int32_t audioSampleRate = 48000;
  const int32_t lowPassFilterDecimation = 10;
  const double maxRfSampleRate = 20e6;
  const double centerFreq = 145e6;
  const double rfSampleRate = round(
      floor(maxRfSampleRate / (audioSampleRate * lowPassFilterDecimation)) * audioSampleRate * lowPassFilterDecimation);

  auto osmoSource = osmosdr::source::make("hackrf=2556b2c3");
  osmoSource->set_sample_rate(rfSampleRate);
  osmoSource->set_center_freq(centerFreq);
  osmoSource->set_gain_mode(false);
  osmoSource->set_gain(0);
  osmoSource->set_dc_offset_mode(osmosdr::source::DCOffsetAutomatic);
  osmoSource->set_iq_balance_mode(osmosdr::source::IQBalanceAutomatic);

  const char* filename = "/home/rick/sdr/raw.bin";

  const auto fileSink = gr::blocks::file_sink::make(8, filename);
  topBlock->connect(osmoSource, 0, fileSink, 0);

  printf("Starting...\n");
  topBlock->start();
  printf("Started\n");

  waitForStop(topBlock);

  printf("Done\n");
  return 0;
}

int nbfmFromRaw() {
  auto topBlock = gr::make_top_block("Top Block");
  runAtExit.emplace([topBlock]() { topBlock->stop(); });

  const int32_t channelCount = 1;
  const int32_t audioSampleRate = 48000;
  const int32_t lowPassFilterDecimation = 10;
  const double maxRfSampleRate = 20e6;
  const double centerFreq = 145e6;
  const double rfSampleRate = round(
      floor(maxRfSampleRate / (audioSampleRate * lowPassFilterDecimation)) * audioSampleRate * lowPassFilterDecimation);
  const gr::blocks::wavfile_format_t waveFormat = gr::blocks::FORMAT_WAV;
  const gr::blocks::wavfile_subformat_t saveSubFormat = gr::blocks::FORMAT_FLOAT;
  const bool append = false;

  const auto source = gr::blocks::file_source::make(8, "/home/rick/sdr/raw.bin", false);

  vector<double> channelFreqs = {
      // 146.46e6,
      145.45e6,
  };
  for (double channelFreq : channelFreqs) {
    char filename[256];
    snprintf(filename, sizeof(filename) - 1, "/home/rick/sdr/%0.3f.wav", channelFreq / 1e6);
    filename[sizeof(filename) - 1] = 0;

    auto wavFileSink =
        gr::blocks::wavfile_sink::make(filename, channelCount, audioSampleRate, waveFormat, saveSubFormat, append);

    FmChannel fmChannel {
        .channelFreq = channelFreq,
        .tau = kTauUs,
        .channelWidth = kNbfmChannelWidth,
    };

    const auto fmDemod = fm_pipeline::make(
        fmChannel,
        centerFreq,
        static_cast<int32_t>(rfSampleRate),
        audioSampleRate,
        lowPassFilterDecimation);

    topBlock->connect(source, 0, fmDemod, 0);
    topBlock->connect(fmDemod, 0, wavFileSink, 0);
  }

  printf("Starting...\n");
  topBlock->start();
  printf("Started\n");

  waitForStop(topBlock);

  printf("Done\n");
  return 0;
}

int nbfm() {
  auto topBlock = gr::make_top_block("Top Block");
  runAtExit.push([topBlock]() { topBlock->stop(); });

  const int32_t channelCount = 1;
  const int32_t audioSampleRate = 48000;
  const int32_t lowPassFilterDecimation = 10;
  const double maxRfSampleRate = 20e6;
  const double centerFreq = 145e6;
  const double rfSampleRate = round(
      floor(maxRfSampleRate / (audioSampleRate * lowPassFilterDecimation)) * audioSampleRate * lowPassFilterDecimation);
  const gr::blocks::wavfile_format_t waveFormat = gr::blocks::FORMAT_WAV;
  const gr::blocks::wavfile_subformat_t saveSubFormat = gr::blocks::FORMAT_FLOAT;
  const bool append = false;

  auto osmoSource = osmosdr::source::make("hackrf=2556b2c3");
  osmoSource->set_sample_rate(rfSampleRate);
  osmoSource->set_center_freq(centerFreq);
  osmoSource->set_gain_mode(false);
  osmoSource->set_gain(0);
  osmoSource->set_dc_offset_mode(osmosdr::source::DCOffsetAutomatic);
  osmoSource->set_iq_balance_mode(osmosdr::source::IQBalanceAutomatic);

  vector<double> channelFreqs = {
      146.46e6,
      145.45e6,
  };
  for (double channelFreq : channelFreqs) {
    char filename[256];
    snprintf(filename, sizeof(filename) - 1, "/home/rick/sdr/%0.3f.wav", channelFreq / 1e6);
    filename[sizeof(filename) - 1] = 0;

    auto wavFileSink =
        gr::blocks::wavfile_sink::make(filename, channelCount, audioSampleRate, waveFormat, saveSubFormat, append);

    FmChannel fmChannel {
        .channelFreq = channelFreq,
        .tau = kTauUs,
        .channelWidth = kNbfmChannelWidth,
    };

    const auto fmDemod = fm_pipeline::make(
        fmChannel,
        centerFreq,
        static_cast<int32_t>(rfSampleRate),
        audioSampleRate,
        lowPassFilterDecimation);

    topBlock->connect(osmoSource, 0, fmDemod, 0);
    topBlock->connect(fmDemod, 0, wavFileSink, 0);
  }

  printf("Starting...\n");
  topBlock->start();
  printf("Started\n");

  waitForStop(topBlock);

  printf("Done\n");
  return 0;
}

int fmRadio(int argc, char** argv) {
  const int32_t channelCount = 1;
  const int32_t audioSampleRate = 48000;
  const int32_t lowPassFilterDecimation = 10;
  const double maxRfSampleRate = 20e6;
  const double centerFreq = 97.8e6;

  const FmChannel krz = {
      .channelFreq = 98.5e6,
      .tau = kTauUs,
      .channelWidth = kWbfmChannelWidth,
  };

  const FmChannel wbRepeater = {
      .channelFreq = 145.45e6,
      .tau = kTauUs,
      .channelWidth = kNbfmChannelWidth,
  };

  const double rfSampleRate = round(
      floor(maxRfSampleRate / (audioSampleRate * lowPassFilterDecimation)) * audioSampleRate * lowPassFilterDecimation);
  auto topBlock = gr::make_top_block("Top Block");
  runAtExit.push([topBlock]() { topBlock->stop(); });

  auto osmoSource = osmosdr::source::make("hackrf=2556b2c3");
  osmoSource->set_sample_rate(rfSampleRate);
  osmoSource->set_center_freq(centerFreq);
  osmoSource->set_gain_mode(false);
  osmoSource->set_gain(0);
  osmoSource->set_dc_offset_mode(osmosdr::source::DCOffsetAutomatic);
  osmoSource->set_iq_balance_mode(osmosdr::source::IQBalanceAutomatic);

  const gr::blocks::wavfile_format_t waveFormat = gr::blocks::FORMAT_WAV;
  const gr::blocks::wavfile_subformat_t saveSubFormat = gr::blocks::FORMAT_FLOAT;
  const bool append = false;

  vector<double> channelFreqs = {
      89.9e6,
      94.3e6,
      94.9e6,
      103.1e6,
  };
  for (double channelFreq : channelFreqs) {
    char filename[256];
    snprintf(filename, sizeof(filename) - 1, "/home/rick/sdr/%0.1f.wav", channelFreq / 1e6);
    filename[sizeof(filename) - 1] = 0;

    auto wavFileSink =
        gr::blocks::wavfile_sink::make(filename, channelCount, audioSampleRate, waveFormat, saveSubFormat, append);

    FmChannel fmChannel {
        .channelFreq = channelFreq,
        .tau = kTauUs,
        .channelWidth = kWbfmChannelWidth,
    };

    const auto fmDemod = fm_pipeline::make(
        fmChannel,
        centerFreq,
        static_cast<int32_t>(rfSampleRate),
        audioSampleRate,
        lowPassFilterDecimation);

    topBlock->connect(osmoSource, 0, fmDemod, 0);
    topBlock->connect(fmDemod, 0, wavFileSink, 0);
  }

  printf("Starting...\n");
  topBlock->start();
  printf("Started\n");

  waitForStop(topBlock);

  printf("Done\n");
  return 0;
}
