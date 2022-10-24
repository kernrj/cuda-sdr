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

#include "HackrfSession.h"

#include <libhackrf/hackrf.h>

#include <mutex>
#include <sstream>

using namespace std;

#define SAFE_HRF(__cmd, __errorMsg)                                     \
  do {                                                                  \
    int status = (__cmd);                                               \
    if (status != HACKRF_SUCCESS) {                                     \
      ostringstream msgStream;                                          \
      msgStream << (__errorMsg) << " - Error " << status << "("         \
                << hackrf_error_name(static_cast<hackrf_error>(status)) \
                << ")";                                                 \
      throw runtime_error(msgStream.str());                             \
    }                                                                   \
  } while (false)

static mutex sessionLock;
static size_t sessionCount = 0;
static bool registeredAtExitHandler = false;

static void cleanupAtExit() {
  lock_guard<mutex> lock(sessionLock);

  if (sessionCount == 0) {
    return;
  }

  sessionCount = 0;
  hackrf_exit();
}

static void incSessionRefCount() {
  lock_guard<mutex> lock(sessionLock);
  if (sessionCount == 0) {
    SAFE_HRF(hackrf_init(), "HackRF library initialization");

    if (!registeredAtExitHandler) {
      int status = std::atexit(cleanupAtExit);

      if (status != 0) {
        fprintf(
            stderr,
            "Failed to register HackRF clean-up. If the program terminates "
            "abnormally, the HackRF library will not be cleanly shutdown and "
            "the HackRF device may need to be reset.\n");
      }
    }
  }

  sessionCount++;
}

static void decSessionRefCount() {
  lock_guard<mutex> lock(sessionLock);

  if (sessionCount == 0) {
    return;
  }

  sessionCount--;

  if (sessionCount == 0) {
    hackrf_exit();
  }
}

HackrfSession::HackrfSession() { incSessionRefCount(); }

HackrfSession::~HackrfSession() { decSessionRefCount(); }
