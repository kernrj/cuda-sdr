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

#include "HackrfSource.h"

#include "ScopeExit.h"

using namespace std;

#define SAFERF(__cmd, __errorMsg)                                                                         \
  do {                                                                                                    \
    int status = (__cmd);                                                                                 \
    if (status != HACKRF_SUCCESS) {                                                                       \
      const string errorName = hackrf_error_name(static_cast<hackrf_error>(status));                      \
      throw runtime_error(string(__errorMsg) + " - Error " + errorName + " (" + to_string(status) + ")"); \
    }                                                                                                     \
  } while (false)

void HackrfSource::setSampleCallback(std::function<void(const int8_t* data, size_t dataLength)>&& sampleCallback) {
  mSampleCallback = std::move(sampleCallback);
}

vector<string> HackrfSource::getDeviceSerialNumbers() {
  hackrf_device_list_t* deviceList = hackrf_device_list();
  const auto freeDeviceList = ScopeExit([deviceList]() { hackrf_device_list_free(deviceList); });

  vector<string> deviceSerialNumbers(deviceList->devicecount);
  for (size_t i = 0; i < deviceList->devicecount; i++) {
    deviceSerialNumbers[i] = deviceList->serial_numbers[i];
  }

  return deviceSerialNumbers;
}

void HackrfSource::selectDeviceByIndex(int deviceIndex) {
  const auto deviceList = shared_ptr<hackrf_device_list_t>(hackrf_device_list(), [](hackrf_device_list_t* deviceList) {
    hackrf_device_list_free(deviceList);
  });

  if (deviceList->devicecount <= 0) {
    throw runtime_error("No HackRF devices found");
  }

  if (deviceIndex >= deviceList->devicecount) {
    throw runtime_error(
        "Device with index [" + to_string(deviceIndex) + "] does not exist. Max is ["
        + to_string(deviceList->devicecount) + "]");
  }

  hackrf_device* deviceRaw = nullptr;
  SAFERF(hackrf_device_list_open(deviceList.get(), deviceIndex, &deviceRaw), "Error opening HackRF device");

  mHackrfDevice = shared_ptr<hackrf_device>(deviceRaw, [](hackrf_device* device) { hackrf_close(device); });

  for (int i = 0; i < deviceList->devicecount; i++) {
    const char* sn = deviceList->serial_numbers[i];
    printf("Device [%d] Serial No [%s]\n", i, sn);
  }
}

void HackrfSource::selectDeviceBySerialNumber(const std::string& serialNumber) {
  const auto deviceList = shared_ptr<hackrf_device_list_t>(hackrf_device_list(), [](hackrf_device_list_t* deviceList) {
    hackrf_device_list_free(deviceList);
  });

  if (deviceList->devicecount <= 0) {
    throw runtime_error("No HackRF devices found");
  }

  int deviceIndex = -1;
  for (int i = 0; i < deviceList->devicecount; i++) {
    string deviceSerialNumber = deviceList->serial_numbers[i];

    if (serialNumber == deviceSerialNumber) {
      deviceIndex = i;
      break;
    }
  }

  if (deviceIndex < 0) {
    throw runtime_error("No device exists with serial number [" + serialNumber + "]");
  }

  hackrf_device* deviceRaw = nullptr;
  SAFERF(hackrf_device_list_open(deviceList.get(), deviceIndex, &deviceRaw), "Error opening HackRF device");

  mHackrfDevice = shared_ptr<hackrf_device>(deviceRaw, [](hackrf_device* device) { hackrf_close(device); });
}

int HackrfSource::rxCallbackWrapper(hackrf_transfer* transfer) {
  auto hackRfSource = reinterpret_cast<HackrfSource*>(transfer->rx_ctx);
  return hackRfSource->rxCallback(transfer);
}

int HackrfSource::rxCallback(hackrf_transfer* transfer) {
  mSampleCallback(reinterpret_cast<const int8_t*>(transfer->buffer), static_cast<size_t>(transfer->valid_length));
  return 0;
}

void HackrfSource::start() {
  if (mHackrfDevice == nullptr) {
    throw runtime_error("Select a HackRF device before starting");
  }

  SAFERF(hackrf_set_freq(mHackrfDevice.get(), 15000000), "Error setting frequency");
  SAFERF(hackrf_set_sample_rate(mHackrfDevice.get(), 256000), "Error setting sample rate");
  SAFERF(hackrf_start_rx(mHackrfDevice.get(), rxCallbackWrapper, this), "Error starting device");
}

void HackrfSource::stop() { SAFERF(hackrf_close(mHackrfDevice.get()), "Cannot close HackRF device"); }
