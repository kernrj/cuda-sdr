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

#include <gpusdrpipeline/GSLog.h>
#include <libhackrf/hackrf.h>

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <functional>
#include <mutex>
#include <stack>
#include <string>

#define SAFERF_RET(__cmd, __errorMsg)                                                                           \
  do {                                                                                                          \
    int status = (__cmd);                                                                                       \
    if (status != HACKRF_SUCCESS) {                                                                             \
      gsloge("%s - Error %s (%d)", (__errorMsg), hackrf_error_name(static_cast<hackrf_error>(status)), status); \
      return 1;                                                                                                 \
    }                                                                                                           \
  } while (false)

using namespace std;

static stack<function<void()>> runAtExit;
static atomic_bool wasInterrupted(false);
static condition_variable wasInterruptedCv;
static mutex shutdownMutex;

static void cleanupThings() {
  while (!runAtExit.empty()) {
    const auto toRun = move(runAtExit.top());
    runAtExit.pop();
    toRun();
  }
}

static void exitSigHandler(int signum) {
  gslogi("Caught signal %d, cleaning up.", signum);
  cleanupThings();

  if (signum == SIGINT || signum == SIGTERM) {
    wasInterrupted.store(true);
    wasInterruptedCv.notify_all();
  } else {
    abort();
  }
}

static int rx_callback(hackrf_transfer* transfer) {
  if (wasInterrupted.load()) {
    return 0;
  }

  gslogi(
      "Buffer size %d %hhd %hhd %hhd %hhd",
      transfer->valid_length,
      transfer->buffer[0],
      transfer->buffer[1],
      transfer->buffer[2],
      transfer->buffer[3]);

  return 0;
}

int main(int argc, char** argv) {
  atexit(cleanupThings);
  signal(SIGSEGV, &exitSigHandler);
  signal(SIGINT, &exitSigHandler);
  signal(SIGTERM, &exitSigHandler);

  SAFERF_RET(hackrf_init(), "Error initializing libhackrf");

  int deviceIndex = 0;

  if (argc > 1) {
    deviceIndex = stoi(argv[1]);
  }

  runAtExit.push([]() {
    gslogi("Cleaning up hackrf");
    hackrf_exit();
  });

  gslogi("HackRF was initialized. Getting device list.");

  hackrf_device_list_t* deviceList = hackrf_device_list();
  gslogi("Device list %p", deviceList);
  gslogi("Device count %d usb %d", deviceList->devicecount, deviceList->usb_devicecount);
  runAtExit.push([deviceList]() {
    gslogi("Freeing device list %p", deviceList);
    hackrf_device_list_free(deviceList);
  });

  if (deviceList->devicecount <= 0) {
    gsloge("No HackRF devices found");
    return ENODEV;
  }

  if (deviceIndex >= deviceList->devicecount) {
    gsloge("Device with index [%d] does not exist. Max is %d.", deviceIndex, deviceList->devicecount);

    return ENODEV;
  }

  hackrf_device* device = nullptr;
  SAFERF_RET(hackrf_device_list_open(deviceList, deviceIndex, &device), "Error opening HackRF device");
  gslogi("Opened device %d", deviceIndex);

  for (int i = 0; i < deviceList->devicecount; i++) {
    const char* sn = deviceList->serial_numbers[i];
    gslogi("Device [%d] Serial No [%s]", i, sn);
  }

  runAtExit.push([device]() {
    gslogi("Closing device %p", device);
    hackrf_close(device);
  });

  SAFERF_RET(hackrf_set_freq(device, 15000000), "Error setting frequency");
  SAFERF_RET(hackrf_set_sample_rate(device, 256000), "Error setting sample rate");
  SAFERF_RET(hackrf_start_rx(device, rx_callback, nullptr), "Error starting device");

  while (!wasInterrupted) {
    unique_lock<mutex> lock(shutdownMutex);
    wasInterruptedCv.wait(lock);
  }

  gslogi("Done");
  return 0;
}
