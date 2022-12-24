//
// Created by Rick Kern on 10/24/22.
//

#include "Window.h"
#include <cmath>

using namespace std;

vector<float> createHammingWindow(size_t length) {
  vector<float> hammingWindow(length);

  for (size_t i = 0; i < length; i++) {
    hammingWindow[i] = 0.54f - 0.46f * cosf(i * 360.0f / (length - 1.0f));
  }

  return hammingWindow;
}
