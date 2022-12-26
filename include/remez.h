//
// Created by Rick Kern on 10/27/22.
//

#ifndef SDRTEST_SRC_REMEZ_H_
#define SDRTEST_SRC_REMEZ_H_

#include <string>
#include <vector>

#define BANDPASS 1
#define DIFFERENTIATOR 2

// type parameter for remez()
enum FilterType { Bandpass, Differentiator };

enum RemezStatus {
  Ok,
  InvalidParameter,
  DidNotConverge,  // Try increasing maxIterations or outTaps size.
  TooManyExtremalFrequencies,
  TooFewExtremalFrequencies,
};

std::string remezStatusToString(RemezStatus status);

struct Band {
  double lowFrequency;
  double highFrequency;
  double lowFrequencyResponse;
  double highFrequencyResponse;
  double weight;
};

/********************
 * Calculates the optimal (in the Chebyshev/minimax sense)
 * FIR filter impulse response given a set of band edges,
 * the desired reponse on those bands, and the weight given to
 * the error in those bands.
 *
 * INPUT:
 * ------
 * int     *numtaps      - Number of filter coefficients
 * int     *numband      - Number of bands in filter specification
 * double  bands[]       - User-specified band edges [2 * numband]
 * double  des[]         - User-specified band responses [2 * numband]
 * double  weight[]      - User-specified error weights [numband]
 * int     *type         - Type of filter
 * int     *griddensity  - Determines how accurately the filter will be
 *                         constructed. The minimum value is 16, but higher
 *                         numbers are slower to compute.
 * int     maxIterations - The number of attempts to refine error before
 *
 * OUTPUT:
 * -------
 * double h[]      - Impulse response of final filter [numtaps]
 *
 * Returns 0 on success. Other values indicate an error.
 */
RemezStatus remezNew(
    std::vector<double>& outTaps,
    const std::vector<Band>& bands,
    FilterType type,
    size_t griddensity,
    size_t maxIterations);

extern "C" void remez(
    double h[],
    int* numtaps,
    int* numband,
    const double bands[],
    const double des[],
    const double weight[],
    int* type,
    int* griddensity);
#endif  // SDRTEST_SRC_REMEZ_H_
