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

#include "HackrfSource.h"

#include <cstring>

#include "GSErrors.h"
#include "util/ScopeExit.h"

using namespace std;

#define SAFERF(hackrfCmd__, errorMsg__)                                                   \
  do {                                                                                    \
    int hackrfStatus = (hackrfCmd__);                                                     \
    if (hackrfStatus != HACKRF_SUCCESS) {                                                 \
      const char* errorName = hackrf_error_name(static_cast<hackrf_error>(hackrfStatus)); \
      gsloge("%s - Error %s (%d)", errorMsg__, errorName, hackrfStatus);                  \
      return Status_RuntimeError;                                                         \
    }                                                                                     \
  } while (false)

const chrono::steady_clock::duration HackrfSource::kDefaultInputTimeout = chrono::seconds(5);

Result<IHackrfSource> HackrfSource::create(
    uint64_t frequency,
    double sampleRate,
    size_t maxBufferCountBeforeDropping,
    IFactories* factories) noexcept {
  Ref<IBufferFactory> sysMemBufferFactory;
  Ref<IBufferPoolFactory> bufferPoolFactory;
  Ref<IBufferCopier> outputBufferCopier;
  Ref<IBufferUtil> bufferUtil;

  UNWRAP_OR_FWD_RESULT(sysMemBufferFactory, factories->createBufferFactory(factories->getSysMemAllocator()));
  UNWRAP_OR_FWD_RESULT(
      bufferPoolFactory,
      factories->createBufferPoolFactory(maxBufferCountBeforeDropping, sysMemBufferFactory.get()));

  vector<ImmutableRef<IBufferCopier>> portOutputCopier = {factories->getSysMemCopier()};

  return makeRefResultNonNull<IHackrfSource>(new (nothrow) HackrfSource(
      frequency,
      sampleRate,
      bufferPoolFactory.get(),
      factories->getSysMemCopier(),
      factories->getBufferUtil(),
      std::move(portOutputCopier)));
}

HackrfSource::HackrfSource(
    uint64_t frequency,
    double sampleRate,
    IBufferPoolFactory* bufferPoolFactory,
    IBufferCopier* outputBufferCopier,
    IBufferUtil* bufferUtil,
    std::vector<ImmutableRef<IBufferCopier>>&& portOutputCopiers) noexcept
    : BaseSource(std::move(portOutputCopiers)),
      mFrequency(frequency),
      mSampleRate(sampleRate),
      mBufferPoolFactory(bufferPoolFactory),
      mOutputBufferCopier(outputBufferCopier),
      mBufferUtil(bufferUtil),
      mInputTimeout(kDefaultInputTimeout),
      mStarted(false) {}

int32_t HackrfSource::getDeviceCount() const noexcept {
  hackrf_device_list_t* deviceList = hackrf_device_list();
  const int32_t deviceCount = deviceList->devicecount;
  hackrf_device_list_free(deviceList);

  return deviceCount;
}

size_t HackrfSource::getDeviceSerialNumber(int32_t deviceIndex, char* buffer, size_t bufferSize) const noexcept {
  hackrf_device_list_t* deviceList = hackrf_device_list();
  const auto freeDeviceList = ScopeExit([deviceList]() { hackrf_device_list_free(deviceList); });

  GS_REQUIRE_OR_RET_FMT(
      deviceIndex < deviceList->devicecount,
      0,
      "Device index [%d] is out of range. Max is [%d]",
      deviceIndex,
      deviceList->devicecount);

  const size_t serialNumberNumChars = strlen(deviceList->serial_numbers[deviceIndex]);
  strncpy(buffer, deviceList->serial_numbers[deviceIndex], bufferSize);

  return serialNumberNumChars;
}

Status HackrfSource::selectDeviceByIndex(int deviceIndex) noexcept {
  const auto deviceList = shared_ptr<hackrf_device_list_t>(hackrf_device_list(), [](hackrf_device_list_t* deviceList) {
    hackrf_device_list_free(deviceList);
  });

  GS_REQUIRE_OR_RET_STATUS(deviceList->devicecount > 0, "No HackRF devices found");
  GS_REQUIRE_OR_RET_STATUS_FMT(
      deviceIndex < deviceList->devicecount,
      "Device with index [%d] does not exist. Max is [%d]",
      deviceIndex,
      deviceList->devicecount);

  hackrf_device* deviceRaw = nullptr;
  SAFERF(hackrf_device_list_open(deviceList.get(), deviceIndex, &deviceRaw), "Error opening HackRF device");

  mHackrfDevice = shared_ptr<hackrf_device>(deviceRaw, [](hackrf_device* device) {
    gslogi("Closing HackRF device");
    hackrf_close(device);
  });

  for (int i = 0; i < deviceList->devicecount; i++) {
    const char* sn = deviceList->serial_numbers[i];
    gslogi("Device [%d] Serial No [%s]", i, sn);
  }

  return Status_Success;
}

Status HackrfSource::selectDeviceBySerialNumber(const char* serialNumber) noexcept {
  const auto deviceList = shared_ptr<hackrf_device_list_t>(hackrf_device_list(), [](hackrf_device_list_t* deviceList) {
    hackrf_device_list_free(deviceList);
  });

  GS_REQUIRE_OR_RET_STATUS(deviceList->devicecount > 0, "No HackRF devices found");

  int deviceIndex = -1;
  for (int i = 0; i < deviceList->devicecount; i++) {
    string deviceSerialNumber = deviceList->serial_numbers[i];

    if (serialNumber == deviceSerialNumber) {
      deviceIndex = i;
      break;
    }
  }

  GS_REQUIRE_OR_RET_STATUS_FMT(deviceIndex >= 0, "No device exists with serial number [%s]", serialNumber);

  hackrf_device* deviceRaw = nullptr;
  SAFERF(hackrf_device_list_open(deviceList.get(), deviceIndex, &deviceRaw), "Error opening HackRF device");

  DO_OR_RET_STATUS(mHackrfDevice = shared_ptr<hackrf_device>(deviceRaw, [](hackrf_device* device) {
                     hackrf_stop_rx(device);
                     hackrf_close(device);
                   }));

  return Status_Success;
}

Status HackrfSource::releaseDevice() noexcept {
  FWD_IF_ERR(stop());
  DO_OR_RET_STATUS(mHackrfDevice.reset());

  return Status_Success;
}

int HackrfSource::rxCallbackWrapper(hackrf_transfer* transfer) noexcept {
  auto hackRfSource = reinterpret_cast<HackrfSource*>(transfer->rx_ctx);
  return hackRfSource->rxCallback(transfer);
}

int HackrfSource::rxCallback(hackrf_transfer* transfer) noexcept CLEANUP_AND_ABORT_ON_EX(
    {
      if (mBufferPool == nullptr || static_cast<int>(mBufferPool->getBufferSize()) < transfer->valid_length) {
        UNWRAP_OR_RETURN(mBufferPool, mBufferPoolFactory->createBufferPool(transfer->valid_length), 0);
      }

      Ref<IBuffer> buffer;
      UNWRAP_OR_RETURN(buffer, mBufferPool->tryGetBuffer(), 0);

      if (buffer == nullptr) {
        gslogi("HackrfSource buffer underrun");
        return 0;
      }

      memmove(buffer->base(), transfer->buffer, transfer->valid_length);
      FWD_IF_ERR(buffer->range()->setUsedRange(0, transfer->valid_length));

      {
        lock_guard lock(mMutex);
        mBuffersAvailableToOutput.push_back(std::move(buffer));
      }

      mOutputBufferAvailable.notify_one();

      return 0;
    },
    releaseDevice());

size_t HackrfSource::getOutputDataSize(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Port [%zu] is out of range", port);

  if (!mStarted) {
    start();
  }

  {
    unique_lock lock(mMutex);

    waitForInputBuffer(lock);

    size_t availableByteCount = 0;
    for (auto& buffer : mBuffersAvailableToOutput) {
      availableByteCount += buffer->range()->used();
    }

    return availableByteCount;
  }
}

size_t HackrfSource::getOutputSizeAlignment(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);

  return 1;
}

Status HackrfSource::readOutput(IBuffer** portOutputBuffers, size_t portCount) noexcept {
  GS_REQUIRE_OR_RET_STATUS(portCount != 0, "One output port is required");

  if (!mStarted) {
    start();
  }

  bool wroteAnyData = false;
  while (portOutputBuffers[0]->range()->hasRemaining()) {
    Ref<IBuffer> buffer;

    {
      unique_lock lock(mMutex);

      if (mBuffersAvailableToOutput.empty() && wroteAnyData) {
        break;
      }

      waitForInputBuffer(lock);

      buffer = mBuffersAvailableToOutput.front();
      mBuffersAvailableToOutput.pop_front();
    }

    FWD_IF_ERR(
        mBufferUtil->moveFromBuffer(portOutputBuffers[0], buffer.get(), buffer->range()->used(), mOutputBufferCopier));

    wroteAnyData = true;
  }

  return Status_Success;
}

Status HackrfSource::start() noexcept {
  if (mStarted) {
    gslogw("HackRF is already started. Ignoring extra call to start()");
    return Status_InvalidState;
  }

  GS_REQUIRE_OR_RET_STATUS(mHackrfDevice != nullptr, "Select a device before starting HackrfSource");

  gslogd("Starting HackRF device");

  SAFERF(hackrf_set_freq(mHackrfDevice.get(), mFrequency), "Error setting frequency");
  SAFERF(hackrf_set_sample_rate(mHackrfDevice.get(), mSampleRate), "Error setting sample rate");
  SAFERF(hackrf_start_rx(mHackrfDevice.get(), rxCallbackWrapper, this), "Error starting device");
  SAFERF(hackrf_set_lna_gain(mHackrfDevice.get(), 0), "Error setting input gain");

  mStarted = true;

  gslogd("Started HackRF device");

  return Status_Success;
}

Status HackrfSource::stop() noexcept {
  if (!mStarted) {
    gslogw("HackRF is already stopped. Ignoring extra call to stop()");
    return Status_InvalidState;
  }

  mHackrfDevice.reset();
  mStarted = false;

  gslogd("Stopped HackRF device");
  cout << "Stopped Hack RF device" << endl;

  return Status_Success;
}

std::chrono::steady_clock::duration HackrfSource::getInputTimeout() const noexcept { return mInputTimeout; }

void HackrfSource::setInputTimeout(const std::chrono::steady_clock::duration& timeout) noexcept {
  mInputTimeout = timeout;
}

Status HackrfSource::waitForInputBuffer(std::unique_lock<std::mutex>& lock) noexcept {
  auto startTime = chrono::steady_clock::now();
  auto endTime = startTime + mInputTimeout;

  while (mBuffersAvailableToOutput.empty()) {
    DO_OR_RET_STATUS(mOutputBufferAvailable.wait_for(lock, endTime - chrono::steady_clock::now()));

    auto now = chrono::steady_clock::now();
    GS_REQUIRE_OR_RET(now <= endTime, "Timed out waiting for HackRF input buffer", Status_TimedOut);
  }

  return Status_Success;
}
