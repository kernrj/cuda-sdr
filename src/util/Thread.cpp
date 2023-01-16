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

#include "util/Thread.h"

using namespace std;

Thread::Thread(function<void()>&& callable)
    : mThreadExited(false),
      mThread([this, threadFnc = std::move(callable)]() {
        threadFnc();

        {
          lock_guard<mutex> lock(mThreadExitedMutex);
          mThreadExited.store(true);
        }
        mThreadExitedCv.notify_all();
      }) {}

void Thread::joinWithTimeout(chrono::microseconds timeout) {
  unique_lock<mutex> lock(mThreadExitedMutex);
  while (!mThreadExited.load()) {
    cv_status status = mThreadExitedCv.wait_for(lock, timeout);

    if (!mThreadExited.load() && status == cv_status::timeout) {
      throw runtime_error("timed out waiting for thread to exit");
    }
  }

  mThread.join();
}
