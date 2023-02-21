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
#include <curl/curl.h>
#include <gpusdrpipeline/GSErrors.h>
#include <mongoose.h>

#include <cstdlib>
#include <iomanip>
#include <list>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>

using namespace std;

#define HEADER_SDP_TYPE "x-sdrgpu-sdp-type"
#define HEADER_SESSION_ID "x-sdrgpu-session-id"
#define HEADER_ICE_CANDIDATE "x-sdrgpu-ice-candidate"

struct IceCandidate {
  Str candidate;
  int64_t sdpMLineIndex;
  Str sdpMid;
  Str usernameFragment;
};

struct SessionDescription {
  string sdp;
  string type;
};

static const mg_str kMethodPost = mg_str("POST");

class SdrSession {
 public:
  explicit SdrSession(const string& sessionId)
      : mSessionId(sessionId) {}

  [[nodiscard]] string getSessionId() const { return mSessionId; }

  void setLocalDescription(const SessionDescription& description) { mLocalDescription = description; }

  void setRemoteDescription(const SessionDescription& description) { mRemoteDescription = description; }

  void addIceCandidate(const IceCandidate& iceCandidate) { mIceCandidates.push_back(iceCandidate); }

 private:
  const string mSessionId;
  SessionDescription mLocalDescription;
  SessionDescription mRemoteDescription;
  std::list<IceCandidate> mIceCandidates;
};

random_device randomDevice;

struct Guid {
  static constexpr size_t PART_COUNT = 4;
  uint32_t parts[PART_COUNT] = {0};

  [[nodiscard]] string toString() const {
    ostringstream out;
    out << setfill('0') << hex;
    out << setw(8) << parts[0] << '-';
    out << setw(4) << (parts[1] >> 16) << '-' << (parts[1] & 0xFFFF) << '-';
    out << (parts[2] >> 16) << '-' << (parts[2] & 0xFFFF);
    out << setw(8) << parts[3];

    return out.str();
  }
};

static Guid createRandomGuid() {
  static_assert(sizeof(Guid::parts[0]) == sizeof(uint32_t));

  Guid guid;
  for (uint32_t& part : guid.parts) {
    part = randomDevice();
  }

  return guid;
}

class ServerContext {
 public:
  [[nodiscard]] shared_ptr<SdrSession> createSession() {
    auto session = make_shared<SdrSession>(createUniqueSessionId());
    mSessions[session->getSessionId()] = session;

    return session;
  }

  [[nodiscard]] bool hasSession(const string& sessionId) const { return mSessions.find(sessionId) != mSessions.end(); }

  [[nodiscard]] shared_ptr<SdrSession> getSession(const string& sessionId) {
    auto it = mSessions.find(sessionId);
    GS_REQUIRE_OR_THROW(it != mSessions.end(), "Unknown session ID [%s]", sessionId.c_str());

    return it->second;
  }

 private:
  unordered_map<string, shared_ptr<SdrSession>> mSessions;

 private:
  [[nodiscard]] string createUniqueSessionId() const {
    string sessionId = createRandomGuid().toString();

    while (mSessions.find(sessionId) != mSessions.end()) {
      sessionId = createRandomGuid().toString();
    }

    return sessionId;
  }
};

/**
 * Returns the total number of bytes used in appendToHeaders, including the null-terminator.
 * If appendToHeaders
 * @param key
 * @param value
 * @param appendToHeaders
 * @param appendToHeadersSize
 * @return
 */
size_t appendHeader(const char* key, const char* value, char* appendToHeaders, size_t appendToHeadersSize) {
  const size_t outputOffset = strlen(appendToHeaders);
  char* newHeaderStart = appendToHeaders + outputOffset;
  const size_t remainingBufferSize = appendToHeadersSize - outputOffset;

  return snprintf(newHeaderStart, remainingBufferSize, "%s: %s\r\n", key, value);
}

char* appendHeaderOrAbort(const char* key, const char* value, char* appendToHeaders, size_t appendToHeadersSize) {
  const size_t nullTerminatorLength = 1;
  size_t size = appendHeader(key, value, appendToHeaders, appendToHeadersSize) + nullTerminatorLength;

  GS_REQUIRE_GTE(
      appendToHeadersSize,
      size,
      "Supplied header buffer size [" << appendToHeadersSize << "] is too small to append the given header. "
                                      << "Key [" << key << "] value [" << value << "] minimum input buffer size ["
                                      << size << "]");

  return appendToHeaders;
}

char* writeHeader(const char* key, const char* value, char* headerBuffer, size_t headerBufferSize) {
  const size_t nullTerminatorLength = 1;
  size_t size = snprintf(headerBuffer, headerBufferSize, "%s: %s\r\n", key, value) + nullTerminatorLength;

  GS_REQUIRE_GTE(
      headerBufferSize,
      size,
      "Supplied header buffer size [" << headerBufferSize << "] is too small for the given header. "
                                      << "Key [" << key << "] value [" << value << "] minimum buffer size [" << size
                                      << "]");

  return headerBuffer;
}

char* createHeader(const char* key, const char* value) {
  static char headerBuffer[1024];
  return writeHeader(key, value, headerBuffer, sizeof(headerBuffer));
}

static void createSession(mg_connection* c, mg_http_message* hm, ServerContext* serverContext) {
  if (mg_strcmp(hm->method, kMethodPost) != 0) {
    mg_http_reply(c, 405, nullptr, "Method must be POST");
    return;
  }

  const auto session = serverContext->createSession();
  const string sessionId = session->getSessionId();

  mg_http_reply(c, 200, createHeader(HEADER_SESSION_ID, sessionId.c_str()), "");
}

string mToString(const mg_str& s) { return string(s.ptr, s.len); }
string mToString(const mg_str* s) { return string(s->ptr, s->len); }

Str mToStr(const mg_str* s) { return IString::create(s->ptr, s->len); }
Str mToStr(const mg_str& s) { return IString::create(s.ptr, s.len); }

static bool getHeader(mg_connection* c, mg_http_message* hm, const char* headerKey, string* headerValueOut) {
  mg_str* headerValue = mg_http_get_header(hm, headerKey);
  if (headerValue == nullptr) {
    mg_http_reply(c, 400, nullptr, "Header [%s] must be set.", headerKey);
    return false;
  }

  *headerValueOut = mToString(headerValue);
  return true;
}

static bool getSession(
    mg_connection* c,
    mg_http_message* hm,
    ServerContext* serverContext,
    shared_ptr<SdrSession>* sessionOut) {
  string sessionId;
  if (!getHeader(c, hm, HEADER_SESSION_ID, &sessionId)) {
    return false;
  }

  if (!serverContext->hasSession(sessionId)) {
    mg_http_reply(c, 400, nullptr, "Session ID [%s] does not exist.", sessionId.c_str());
    return false;
  }

  *sessionOut = serverContext->getSession(sessionId);
  return true;
}

static void setAnswer(mg_connection* c, mg_http_message* hm, ServerContext* serverContext) {
  shared_ptr<SdrSession> session;
  if (!getSession(c, hm, serverContext, &session)) {
    return;
  }

  string sessionId;
  string sdpType;
  if (!getHeader(c, hm, HEADER_SDP_TYPE, &sdpType)) {
    return;
  }

  session->setRemoteDescription(SessionDescription {
      .sdp = mToString(hm->body),
      .type = sdpType,
  });
}

static void addIceCandidate(mg_connection* c, mg_http_message* hm, ServerContext* serverContext) {
  shared_ptr<SdrSession> session;
  if (!getSession(c, hm, serverContext, &session)) {
    return;
  }

  string iceCandidate;
  if (!getHeader(c, hm, HEADER_ICE_CANDIDATE, &iceCandidate)) {
    return;
  }

  // session->addIceCandidate()
}

static void handleHttpMessage(mg_connection* c, mg_http_message* hm, void* ctx) {
  ServerContext* const serverContext = reinterpret_cast<ServerContext*>(ctx);

  if (mg_http_match_uri(hm, "/createSession")) {
    createSession(c, hm, serverContext);
  } else if (mg_http_match_uri(hm, "/setAnswer")) {
    setAnswer(c, hm, serverContext);
  } else if (mg_http_match_uri(hm, "/addIceCandidate")) {
    addIceCandidate(c, hm, serverContext);
  } else {
    mg_http_reply(c, 404, "Not Found", "Path not found");
  }
}

static void mgEventHandler(mg_connection* c, int ev, void* ev_data, void* fn_data) {
  if (ev == MG_EV_HTTP_MSG) {
    auto hm = reinterpret_cast<mg_http_message*>(ev_data);

    handleHttpMessage(c, hm, fn_data);
  }
}

int main(int argc, char** argv) {
  mg_mgr mgr;
  mg_mgr_init(&mgr);
  ServerContext context;
  mg_connection* connection = mg_http_listen(&mgr, "0.0.0.0:8000", mgEventHandler, &context);
  for (;;) {
    mg_mgr_poll(&mgr, 1000);
  }
  return 0;
}
