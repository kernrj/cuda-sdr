/*
 * Copyright 2022-2023 Rick Kern <kernrj@gmail.com>
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

#ifndef SDRTEST_SRC_HACKRFSOURCE_H_
#define SDRTEST_SRC_HACKRFSOURCE_H_

#include <libhackrf/hackrf.h>

#include <complex>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "Factories.h"
#include "HackrfSession.h"
#include "filters/BaseSource.h"
#include "filters/IHackrfSource.h"

class HackrfSource final : public IHackrfSource, public BaseSource {
 public:
  static Result<IHackrfSource> create(
      uint64_t frequency,
      double sampleRate,
      size_t maxBufferCountBeforeDropping,
      IFactories* factories) noexcept;

  int32_t getDeviceCount() const noexcept final;
  size_t getDeviceSerialNumber(int32_t deviceIndex, char* buffer, size_t bufferSize) const noexcept final;

  Status selectDeviceByIndex(int deviceIndex) noexcept final;
  Status selectDeviceBySerialNumber(const char* serialNumber) noexcept final;
  Status releaseDevice() noexcept final;

  Status start() noexcept final;
  Status stop() noexcept final;

  size_t getOutputDataSize(size_t port) noexcept final;
  size_t getOutputSizeAlignment(size_t port) noexcept final;
  Status readOutput(IBuffer** portOutputBuffers, size_t numPorts) noexcept final;

  std::chrono::steady_clock::duration getInputTimeout() const noexcept;
  void setInputTimeout(const std::chrono::steady_clock::duration& timeout) noexcept;

 private:
  static const std::chrono::steady_clock::duration kDefaultInputTimeout;
  HackrfSession mHackrfSession;

  const uint64_t mFrequency;
  const double mSampleRate;
  ConstRef<IBufferPoolFactory> mBufferPoolFactory;
  ConstRef<IBufferCopier> mOutputBufferCopier;
  ConstRef<IBufferUtil> mBufferUtil;
  std::chrono::steady_clock::duration mInputTimeout;

  std::mutex mMutex;
  std::condition_variable mOutputBufferAvailable;

  Ref<IBufferPool> mBufferPool;
  std::shared_ptr<hackrf_device> mHackrfDevice;
  std::list<ImmutableRef<IBuffer>> mBuffersAvailableToOutput;
  bool mStarted;

 private:
  HackrfSource(
      uint64_t frequency,
      double sampleRate,
      IBufferPoolFactory* bufferPoolFactory,
      IBufferCopier* outputBufferCopier,
      IBufferUtil* bufferUtil,
      std::vector<ImmutableRef<IBufferCopier>>&& portOutputCopiers) noexcept;

  static int rxCallbackWrapper(hackrf_transfer* transfer) noexcept;
  int rxCallback(hackrf_transfer* transfer) noexcept;
  Status waitForInputBuffer(std::unique_lock<std::mutex>& lock) noexcept;

  REF_COUNTED(HackrfSource);
};

#endif  // SDRTEST_SRC_HACKRFSOURCE_H_
