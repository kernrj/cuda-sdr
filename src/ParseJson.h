/*
 * Copyright 2023 Rick Kern <kernrj@gmail.com>
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

#ifndef GPUSDRPIPELINE_PARSEJSON_H
#define GPUSDRPIPELINE_PARSEJSON_H

#include <nlohmann/json.hpp>
#include <string>

#include "Result.h"
#include "SampleType.h"


/**
 * Parse JSON and return the parsed json, or a status code on error. No exceptions are thrown.
 *
 * @param jsonString The JSON string to parse.
 * @param requiredType If nlohmann::json::value_t::null, the type is not checked.
 * @return A Result with its value set, and its status set to Status_Success on success.
 *         Its status will be set to an error code otherwise.
 */
Result<nlohmann::json> parseJson(const char* jsonString, nlohmann::json::value_t requiredType) noexcept;
Result<nlohmann::json> parseJson(const char* jsonString) noexcept;

template <typename T>
Result<T> getJsonOrErr(const nlohmann::json& obj, const char* key) {
  if (!obj.contains(key)) {
    gsloge("Key [%s] is not present", key);
    return ERR_RESULT(Status_NotFound);
  }

  try {
    return makeValResult<T>(obj[key].get<T>());
  } catch (const std::exception& e) {
    gslogw("Failed to get key [%s]. Error [%s]", key, e.what());
    return ERR_RESULT(Status_UnknownError);
  } catch (...) {
    gslogw("Failed to get key [%s]. Caught non-exception.", key);
    return ERR_RESULT(Status_UnknownError);
  }
}

template <typename T>
Result<T> getJsonOrErr(const nlohmann::json& obj, const std::string& key) {
  return getJsonOrErr<T>(obj, key.c_str());
}

template <typename T>
Result<T> getJsonOrErr(const nlohmann::json& obj, const nlohmann::json_pointer<std::string>& key) {
  if (!obj.contains(key)) {
    gsloge("Pointer key [%s] is not present", key.to_string().c_str());
    return ERR_RESULT(Status_NotFound);
  }

  try {
    return makeValResult<T>(obj[key].get<T>());
  } catch (const std::exception& e) {
    gslogw("Failed to get pointer key [%s]. Error [%s]", key, e.what());
    return ERR_RESULT(Status_UnknownError);
  } catch (...) {
    gslogw("Failed to get pointer key [%s]. Caught non-exception.", key);
    return ERR_RESULT(Status_UnknownError);
  }
}

template <typename T>
T tryGetJson(const nlohmann::json& obj, const char* key, const T& valueIfNotFound) noexcept {
  if (!obj.contains(key)) {
    return valueIfNotFound;
  }

  const nlohmann::json& j = obj[key];

  try {
    return j.get<T>();
  } catch (const std::exception& e) {
    gslogw("Failed to get key [%s]. Error [%s]", key, e.what());
    return valueIfNotFound;
  }
}

template <typename T>
T tryGetJson(const nlohmann::json& obj, const char* key) noexcept {
  if (!obj.contains(key)) {
    return {};
  }

  const nlohmann::json& j = obj[key];

  try {
    return j.get<T>();
  } catch (const std::exception& e) {
    gslogw("Failed to get key [%s]. Error [%s]", key, e.what());
    return {};
  }
}

template <typename T>
T getJson(const nlohmann::json& obj, const char* key) {
  if (!obj.contains(key)) {
    gsloge("Key [%s] is not present", key);
    throw std::runtime_error(std::string("Key [") + key + "] is not present");
  }

  const nlohmann::json& j = obj[key];

  try {
    return j.get<T>();
  } catch (const std::exception& e) {
    gslogw("Failed to get key [%s]. Error [%s]", key, e.what());
    return {};
  }
}

template <typename T>
T getJson(const nlohmann::json& obj, const std::string key) {
  return getJson<T>(obj, key.c_str());
}

template <typename T>
T getJson(const nlohmann::json& obj, const nlohmann::json_pointer<std::string>& key) {
  if (!obj.contains(key)) {
    gsloge("Key [%s] is not present", key.to_string().c_str());
    throw std::runtime_error(std::string("Key [") + key.to_string() + "] is not present");
  }

  const nlohmann::json& j = obj[key];

  try {
    return j.get<T>();
  } catch (const std::exception& e) {
    gslogw("Failed to get key [%s]. Error [%s]", key, e.what());
    return {};
  }
}

inline std::unordered_map<std::string, nlohmann::json> tryGetJsonObj(
    const nlohmann::json& obj,
    const char* key) noexcept {
  return tryGetJson<std::unordered_map<std::string, nlohmann::json>>(obj, key);
}

inline std::vector<nlohmann::json> tryGetJsonArray(const nlohmann::json& obj, const char* key) noexcept {
  return tryGetJson<std::vector<nlohmann::json>>(obj, key);
}

inline Result<SampleType> parseSampleType(const std::string& sampleTypeStr) noexcept {
  struct SampleTypeMapping {
    const SampleType sampleType;
    const std::string sampleTypeName;
  };

  static std::vector<SampleTypeMapping> sampleTypes = {
      {.sampleType = SampleType_FloatComplex, .sampleTypeName = "FloatComplex"},
      {.sampleType = SampleType_Float, .sampleTypeName = "Float"},
      {.sampleType = SampleType_Int8Complex, .sampleTypeName = "Int8Complex"},
  };

  for (const SampleTypeMapping& mapping : sampleTypes) {
    if (sampleTypeStr == mapping.sampleTypeName) {
      return makeValResult(mapping.sampleType);
    }
  }

  gsloge("Unknown sample type [%s]", sampleTypeStr.c_str());
  return ERR_RESULT(Status_ParseError);
}

#endif  // GPUSDRPIPELINE_PARSEJSON_H
