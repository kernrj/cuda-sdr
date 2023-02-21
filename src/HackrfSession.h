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

#ifndef SDRTEST_SRC_HACKRFSESSION_H_
#define SDRTEST_SRC_HACKRFSESSION_H_

/**
 * When the number of allocated HackrfSessions increases from 0 to 1, the
 * HackRF library will be initialized.
 *
 * Similarly, when the number of allocated HackrfSessions decreases from 1 to 0,
 * the HackRF library will be uninitialized.
 */
class HackrfSession {
 public:
  HackrfSession();
  ~HackrfSession();
};

#endif  // SDRTEST_SRC_HACKRFSESSION_H_
