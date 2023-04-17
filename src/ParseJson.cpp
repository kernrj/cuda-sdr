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

#include "ParseJson.h"

#include <cinttypes>

#include "GSErrors.h"

using namespace std;
using namespace nlohmann;

Result<json> parseJson(const char* jsonString) noexcept {
  try {
    return makeValResult(nlohmann::json::parse(jsonString));
  } catch (const json::parse_error& e) {
    gsloge("Failed to parse [%s] %s", e.what(), jsonString);
    return ERR_RESULT(Status_ParseError);
  }
  IF_CATCH_RETURN_RESULT;
}

Result<json> parseJson(const char* jsonString, json::value_t requiredType) noexcept {
  Result<json> parsedJsonResult = parseJson(jsonString);

  if (requiredType != json::value_t::null && parsedJsonResult.value.type() != requiredType) {
    const char* typeName;

    try {
      typeName = json(requiredType).type_name();
    } catch (...) {
      typeName = "<Error getting type string>";
    }

    gsloge("JSON parse error: expected to parse [%s], but got [%s]",
           typeName,
           parsedJsonResult.value.type_name());

    return ERR_RESULT(Status_ParseError);
  }

  return parsedJsonResult;
}

Result<string> getStringFromJson(const json& obj, const char* key) noexcept {
  GS_REQUIRE_OR_RET_RESULT_FMT(obj.contains(key), "Required key [%s] is missing", key);
  const nlohmann::json& j = obj[key];
  GS_REQUIRE_OR_RET_RESULT_FMT(j.is_string(), "Value for [%s] must be a string", key);

  DO_OR_RET_ERR_RESULT(return makeValResult(j.get<string>()));
}

Result<int64_t> getIntFromJson(const json& obj, const char* key) noexcept {
  GS_REQUIRE_OR_RET_RESULT_FMT(obj.contains(key), "Required key [%s] is missing", key);
  const nlohmann::json& j = obj[key];
  GS_REQUIRE_OR_RET_RESULT_FMT(j.is_number_integer(), "Value for [%s] must be an integer", key);

  DO_OR_RET_ERR_RESULT(return makeValResult(j.get<int64_t>()));
}

string getStringFromJsonOr(const json& obj, const char* key, const char* valueIfNotFound) noexcept {
  if (!obj.contains(key)) {
    return valueIfNotFound;
  }

  const json& j = obj[key];
  if (!j.is_string()) {
    gslogw(
        "Key [%s] is type [%s], but an integer is expected. Returning default value [%s].",
        key,
        j.type_name(),
        valueIfNotFound);
    return valueIfNotFound;
  }

  try {
    return j.get<string>();
  } catch(const exception& e) {
    gslogw("Failed to get key [%s] as a string. Error [%s]", key, e.what());
    return valueIfNotFound;
  }
}

int64_t getIntFromJsonOr(const nlohmann::json& obj, const char* key, int64_t valueIfNotFound) noexcept {
  if (!obj.contains(key)) {
    return valueIfNotFound;
  }

  const nlohmann::json& j = obj[key];
  if (!j.is_number_integer()) {
    gslogw(
        "Key [%s] is type [%s], but an integer is expected. Returning default value %" PRId64 ".",
        key,
        j.type_name(),
        valueIfNotFound);
    return valueIfNotFound;
  }

  try {
    return j.get<int64_t>();
  } catch(const exception& e) {
    gslogw("Failed to get key [%s] as an int64_t. Error [%s]", key, e.what());
    return valueIfNotFound;
  }
}
