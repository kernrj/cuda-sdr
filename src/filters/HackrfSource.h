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
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "Factories.h"
#include "HackrfSession.h"
#include "filters/IHackrfSource.h"

class HackrfSource : public IHackrfSource {
 public:
  HackrfSource(uint64_t frequency, double sampleRate, size_t maxBufferCountBeforeDropping, IFactories* factories);

  std::vector<std::string> getDeviceSerialNumbers() override;

  void selectDeviceByIndex(int deviceIndex) override;
  void selectDeviceBySerialNumber(const std::string& serialNumber) override;

  void start() override;
  void stop() override;

  size_t getOutputDataSize(size_t port) override;
  size_t getOutputSizeAlignment(size_t port) override;
  void readOutput(const std::vector<std::shared_ptr<IBuffer>>& portOutputs) override;

 private:
  HackrfSession mHackrfSession;

  const uint64_t mFrequency;
  const double mSampleRate;
  const std::shared_ptr<IBufferPoolFactory> mBufferPoolFactory;
  const std::shared_ptr<IBufferCopier> mOutputBufferCopier;
  const std::shared_ptr<IBufferUtil> mBufferUtil;

  std::mutex mMutex;
  std::condition_variable mOutputBufferAvailable;

  std::shared_ptr<IBufferPool> mBufferPool;
  std::shared_ptr<hackrf_device> mHackrfDevice;
  std::queue<std::shared_ptr<IBuffer>> mBuffersAvailableToOutput;
  bool mStarted;

 private:
  static int rxCallbackWrapper(hackrf_transfer* transfer);
  int rxCallback(hackrf_transfer* transfer);
};

#endif  // SDRTEST_SRC_HACKRFSOURCE_H_
