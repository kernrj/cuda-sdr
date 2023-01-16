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

#ifndef SDRTEST_SRC_IHACKRFSOURCE_H_
#define SDRTEST_SRC_IHACKRFSOURCE_H_

#include <gpusdrpipeline/filters/Filter.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

class IHackrfSource : public Source {
 public:
  virtual std::vector<std::string> getDeviceSerialNumbers() = 0;

  virtual void selectDeviceByIndex(int32_t deviceIndex) = 0;
  virtual void selectDeviceBySerialNumber(const std::string& serialNumber) = 0;

  virtual void start() = 0;
  virtual void stop() = 0;
};

#endif  // SDRTEST_SRC_IHACKRFSOURCE_H_
