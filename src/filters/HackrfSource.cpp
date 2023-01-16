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

#include <cstring>

#include "GSErrors.h"
#include "util/ScopeExit.h"

using namespace std;

#define SAFERF(__cmd, __errorMsg)                                                                         \
  do {                                                                                                    \
    int status = (__cmd);                                                                                 \
    if (status != HACKRF_SUCCESS) {                                                                       \
      const string errorName = hackrf_error_name(static_cast<hackrf_error>(status));                      \
      throw runtime_error(string(__errorMsg) + " - Error " + errorName + " (" + to_string(status) + ")"); \
    }                                                                                                     \
  } while (false)

HackrfSource::HackrfSource(
    uint64_t frequency,
    double sampleRate,
    size_t maxBufferCountBeforeDropping,
    IFactories* factories)
    : mFrequency(frequency),
      mSampleRate(sampleRate),
      mBufferPoolFactory(factories->createBufferPoolFactory(
          maxBufferCountBeforeDropping,
          factories->getBufferFactory(factories->getSysMemAllocator()))),
      mOutputBufferCopier(factories->getSysMemCopier()),
      mBufferUtil(factories->getBufferUtil()),
      mStarted(false) {}

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

void HackrfSource::selectDeviceBySerialNumber(const string& serialNumber) {
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

  mHackrfDevice = shared_ptr<hackrf_device>(deviceRaw, [](hackrf_device* device) {
    hackrf_stop_rx(device);
    hackrf_close(device);
  });
}

int HackrfSource::rxCallbackWrapper(hackrf_transfer* transfer) {
  auto hackRfSource = reinterpret_cast<HackrfSource*>(transfer->rx_ctx);
  return hackRfSource->rxCallback(transfer);
}

int HackrfSource::rxCallback(hackrf_transfer* transfer) {
  if (mBufferPool == nullptr || mBufferPool->getBufferSize() < transfer->valid_length) {
    mBufferPool = mBufferPoolFactory->createBufferPool(transfer->valid_length);
  }

  optional<shared_ptr<IBuffer>> bufferOpt = mBufferPool->tryGetBuffer();
  if (!bufferOpt.has_value()) {
    printf("HackrfSource buffer underrun\n");
    return 0;
  }

  auto& buffer = bufferOpt.value();
  memmove(buffer->base(), transfer->buffer, transfer->valid_length);
  buffer->range()->setUsedRange(0, transfer->valid_length);

  {
    lock_guard lock(mMutex);
    mBuffersAvailableToOutput.push(std::move(buffer));
  }

  mOutputBufferAvailable.notify_one();

  return 0;
}

size_t HackrfSource::getOutputDataSize(size_t port) {
  if (port != 0) {
    THROW("Unexpected port [" << port << "]. Max value is 0");
  } else if (!mStarted) {
    THROW("The HackRF source must be started before requesting the output size");
  }

  {
    unique_lock lock(mMutex);

    while (mBuffersAvailableToOutput.empty()) {
      mOutputBufferAvailable.wait(lock);
    }

    return mBuffersAvailableToOutput.front()->range()->used();
  }
}

size_t HackrfSource::getOutputSizeAlignment(size_t port) {
  if (port != 0) {
    THROW("Unexpected port [" << port << "]. Max value is 0");
  }

  return 1;
}

void HackrfSource::readOutput(const vector<shared_ptr<IBuffer>>& portOutputs) {
  if (portOutputs.empty()) {
    THROW("One output port is required");
  }

  if (!mStarted) {
    THROW("The HackRF source must be started before requesting output");
  }

  shared_ptr<IBuffer> buffer;

  {
    unique_lock lock(mMutex);

    while (mBuffersAvailableToOutput.empty()) {
      mOutputBufferAvailable.wait(lock);
    }

    buffer = mBuffersAvailableToOutput.front();
    mBuffersAvailableToOutput.pop();
  }

  mBufferUtil->moveFromBuffer(portOutputs[0], buffer, buffer->range()->used(), mOutputBufferCopier);
}

void HackrfSource::start() {
  if (mStarted) {
    return;
  }

  if (mHackrfDevice == nullptr) {
    throw runtime_error("Select a HackRF device before starting");
  }

  SAFERF(hackrf_set_freq(mHackrfDevice.get(), mFrequency), "Error setting frequency");
  SAFERF(hackrf_set_sample_rate(mHackrfDevice.get(), mSampleRate), "Error setting sample rate");
  SAFERF(hackrf_start_rx(mHackrfDevice.get(), rxCallbackWrapper, this), "Error starting device");

  mStarted = true;
}

void HackrfSource::stop() {
  mHackrfDevice.reset();
  mStarted = false;
}
