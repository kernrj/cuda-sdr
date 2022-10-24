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

#ifndef SDRTEST_SRC_HACKRFSOURCE_H_
#define SDRTEST_SRC_HACKRFSOURCE_H_

#include <libhackrf/hackrf.h>

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "HackrfSession.h"

class HackrfSource {
 public:
  void selectDeviceByIndex(int deviceIndex);
  void selectDeviceBySerialNumber(const std::string& serialNumber);

  std::vector<std::string> getDeviceSerialNumbers();

  void start();
  void stop();

  void setSampleCallback(
      std::function<void(const int8_t* data, size_t dataLength)>&& sampleCallback);

 private:
  HackrfSession mHackrfSession;
  std::shared_ptr<hackrf_device> mHackrfDevice;
  std::function<void(const int8_t* data, size_t dataLength)> mSampleCallback;

 private:
  static int rxCallbackWrapper(hackrf_transfer* transfer);
  int rxCallback(hackrf_transfer* transfer);
};

#endif  // SDRTEST_SRC_HACKRFSOURCE_H_
