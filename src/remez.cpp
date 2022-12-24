/**************************************************************************
 * Parks-McClellan algorithm for FIR filter design (C version)
 *-------------------------------------------------
 *  Copyright (c) 1995,1998  Jake Janovetz <janovetz@uiuc.edu>
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Library General Public
 *  License as published by the Free Software Foundation; either
 *  version 2 of the License, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Library General Public License for more details.
 *
 *  You should have received a copy of the GNU Library General Public
 *  License along with this library; if not, write to the Free
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *
 *  Sep 1999 - Paul Kienzle (pkienzle@cs.indiana.edu)
 *      Modified for use in octave as a replacement for the matlab function
 *      remez.mex.  In particular, magnitude responses are required for all
 *      band edges rather than one per band, griddensity is a parameter,
 *      and errors are returned rather than printed directly.
 *  Mar 2000 - Kai Habel (kahacjde@linux.zrz.tu-berlin.de)
 *      Change: ColumnVector x=arg(i).vector_value();
 *      to: ColumnVector x(arg(i).vector_value());
 *  There appear to be some problems with the routine Search. See comments
 *  therein [search for PAK:].  I haven't looked closely at the rest
 *  of the code---it may also have some problems.
 *************************************************************************/

#include "remez.h"

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using namespace std;

enum Symmetry {
  NEGATIVE,
  POSITIVE,
};

#define Pi 3.1415926535897932
#define Pi2 6.2831853071795865

std::string remezStatusToString(RemezStatus status) {
  switch (status) {
    case Ok:
      return "Success";
    case InvalidParameter:
      return "Invalid Parameter";
    case TooManyExtremalFrequencies:
      return "Too many extremal frequencies";
    case TooFewExtremalFrequencies:
      return "Too few extremal frequencies";
    case DidNotConverge:
      return "Did not converge";
    default:
      return "Unknown (" + to_string(status) + ")";
  }
}

/*******************
 * CreateDenseGrid
 *=================
 * Creates the dense grid of frequencies from the specified bands.
 * Also creates the Desired Frequency Response function (D[]) and
 * the Weight function (W[]) on that dense grid
 *
 *
 * INPUT:
 * ------
 * size_t      r        - 1/2 the number of filter coefficients
 * size_t      numtaps  - Number of taps in the resulting filter
 * double   bands[]  - User-specified band edges [2*numband]
 * double   des[]    - Desired response per band [2*numband]
 * double   weight[] - Weight per band [numband]
 * size_t      symmetry - Symmetry of filter - used for grid check
 * size_t      griddensity
 *
 * OUTPUT:
 * -------
 * double Grid[]     - Frequencies (0 to 0.5) on the dense grid [gridsize]
 * double D[]        - Desired response on the dense grid [gridsize]
 * double W[]        - Weight function on the dense grid [gridsize]
 *******************/
[[nodiscard]] static RemezStatus CreateDenseGrid(
    size_t r,
    size_t numtaps,
    const vector<Band>& bands,
    vector<double>& Grid,
    vector<double>& D,
    vector<double>& W,
    Symmetry symmetry,
    size_t griddensity) {
  if (Grid.size() != D.size() || Grid.size() != W.size()) {
    return InvalidParameter;
  }

  const double delf = 0.5 / static_cast<double>((griddensity * r));

  /*
   * For differentiator, hilbert,
   *   symmetry is odd and Grid[0] = max(delf, bands[0])
   */
  const double grid0 =
      ((symmetry == NEGATIVE) && (delf > bands[0].lowFrequency))
          ? delf
          : bands[0].lowFrequency;

  for (size_t j = 0, bandIndex = 0; bandIndex < bands.size(); bandIndex++) {
    const Band& band = bands[bandIndex];
    double lowf = (bandIndex == 0 ? grid0 : band.lowFrequency);
    const double highf = band.highFrequency;
    const size_t k = lrint((highf - lowf) / delf);

    for (size_t i = 0; i < k; i++) {
      if (j < Grid.size()) {
        D[j] = band.lowFrequencyResponse
               + static_cast<double>(i)
                     * (band.highFrequencyResponse - band.lowFrequencyResponse)
                     / static_cast<double>(k - 1);
        W[j] = band.weight;
        Grid[j] = lowf;
        lowf += delf;
        j++;
      }
    }
    if (j - 1 < Grid.size()) {
      Grid[j - 1] = highf;
    }

    if (j > Grid.size()) {
      fprintf(stderr, "Grid size is too small. At least [%zd] is needed.\n", k);
      return TooFewExtremalFrequencies;
    }
  }

  /*
   * Similar to above, if odd symmetry, last grid point can't be .5
   *  - but, if there are even taps, leave the last grid point at .5
   */
  if ((symmetry == NEGATIVE) && (Grid[Grid.size() - 1] > (0.5 - delf))
      && (numtaps % 2)) {
    Grid[Grid.size() - 1] = 0.5 - delf;
  }

  return Ok;
}

/********************
 * InitialGuess
 *==============
 * Places Extremal Frequencies evenly throughout the dense grid.
 *
 *
 * INPUT:
 * ------
 * size_t r        - 1/2 the number of filter coefficients
 * size_t gridsize - Number of elements in the dense frequency grid
 *
 * OUTPUT:
 * -------
 * size_t Ext[]    - Extremal indexes to dense frequency grid [r+1]
 ********************/

[[nodiscard]] static RemezStatus InitialGuess(
    vector<size_t>& Ext,
    size_t gridsize) {
  if (gridsize == 0) {
    return InvalidParameter;
  }

  const size_t r = Ext.size() - 1;
  for (size_t i = 0; i < Ext.size(); i++) {
    Ext[i] = i * (gridsize - 1) / r;
  }

  return Ok;
}

/***********************
 * CalcParms
 *===========
 *
 *
 * INPUT:
 * ------
 * size_t    Ext[]  - Extremal indexes to dense frequency grid [r+1]
 * double Grid[] - Frequencies (0 to 0.5) on the dense grid [gridsize]
 * double D[]    - Desired response on the dense grid [gridsize]
 * double W[]    - Weight function on the dense grid [gridsize]
 *
 * OUTPUT:
 * -------
 * double ad[]   - 'b' in Oppenheim & Schafer [r+1]
 * double x[]    - [r+1]
 * double y[]    - 'C' in Oppenheim & Schafer [r+1]
 ***********************/
[[nodiscard]] static RemezStatus CalcParms(
    const vector<size_t>& Ext,
    const vector<double>& Grid,
    const vector<double>& D,
    const vector<double>& W,
    vector<double>& ad,
    vector<double>& x,
    vector<double>& y) {
  if (Ext.size() != ad.size() || Ext.size() != x.size()
      || Ext.size() != y.size()) {
    return InvalidParameter;
  }

  if (Grid.size() != D.size() || Grid.size() != W.size()) {
    return InvalidParameter;
  }

  const size_t r = Ext.size() - 1;

  /*
   * Find x[]
   */
  for (size_t i = 0; i <= r; i++) {
    x[i] = cos(Pi2 * Grid[Ext[i]]);
  }

  /*
   * Calculate ad[]  - Oppenheim & Schafer eq 7.132
   */
  size_t ld = (r - 1) / 15 + 1; /* Skips around to avoid round errors */
  for (size_t i = 0; i <= r; i++) {
    double denom = 1.0;
    double xi = x[i];
    for (size_t j = 0; j < ld; j++) {
      for (size_t k = j; k <= r; k += ld)
        if (k != i) denom *= 2.0 * (xi - x[k]);
    }
    if (fabs(denom) < 0.00001) denom = 0.00001;
    ad[i] = 1.0 / denom;
  }

  /*
   * Calculate delta  - Oppenheim & Schafer eq 7.131
   */
  double numer = 0.0;
  double denom = 0.0;
  int32_t sign = 1;
  for (size_t i = 0; i <= r; i++) {
    numer += ad[i] * D[Ext[i]];
    denom += sign * ad[i] / W[Ext[i]];
    sign = -sign;
  }
  const double delta = numer / denom;
  sign = 1;

  /*
   * Calculate y[]  - Oppenheim & Schafer eq 7.133b
   */
  for (size_t i = 0; i <= r; i++) {
    y[i] = D[Ext[i]] - sign * delta / W[Ext[i]];
    sign = -sign;
  }

  return Ok;
}

/*********************
 * ComputeA
 *==========
 * Using values calculated in CalcParms, ComputeA calculates the
 * actual filter response at a given frequency (freq).  Uses
 * eq 7.133a from Oppenheim & Schafer.
 *
 *
 * INPUT:
 * ------
 * double freq - Frequency (0 to 0.5) at which to calculate A
 * size_t    r    - 1/2 the number of filter coefficients
 * double ad[] - 'b' in Oppenheim & Schafer [r+1]
 * double x[]  - [r+1]
 * double y[]  - 'C' in Oppenheim & Schafer [r+1]
 *
 * OUTPUT:
 * -------
 * Returns double value of A[freq]
 *********************/

[[nodiscard]] static RemezStatus ComputeA(
    double freq,
    const vector<double>& ad,
    const vector<double>& x,
    const vector<double>& y,
    double* outA) {
  const size_t r = ad.size() - 1;

  if (ad.size() != x.size() || ad.size() != y.size() || outA == nullptr) {
    return InvalidParameter;
  }

  double numer = 0.0;
  double denom = 0.0;
  const double xc = cos(Pi2 * freq);
  for (size_t i = 0; i <= r; i++) {
    double c = xc - x[i];
    if (fabs(c) < 1.0e-7) {
      numer = y[i];
      denom = 1;
      break;
    }
    c = ad[i] / c;
    denom += c;
    numer += c * y[i];
  }

  *outA = numer / denom;

  return Ok;
}

/************************
 * CalcError
 *===========
 * Calculates the Error function from the desired frequency response
 * on the dense grid (D[]), the weight function on the dense grid (W[]),
 * and the present response calculation (A[])
 *
 *
 * INPUT:
 * ------
 * size_t    r      - 1/2 the number of filter coefficients
 * double ad[]   - [r+1]
 * double x[]    - [r+1]
 * double y[]    - [r+1]
 * double Grid[] - Frequencies on the dense grid [gridsize]
 * double D[]    - Desired response on the dense grid [gridsize]
 * double W[]    - Weight function on the desnse grid [gridsize]
 *
 * OUTPUT:
 * -------
 * double E[]    - Error function on dense grid [gridsize]
 ************************/
[[nodiscard]] static RemezStatus CalcError(
    const vector<double>& ad,
    const vector<double>& x,
    const vector<double>& y,
    const vector<double>& Grid,
    const vector<double>& D,
    const vector<double>& W,
    vector<double>& E) {
  if (ad.size() != x.size() || ad.size() != y.size()) {
    return InvalidParameter;
  }

  if (Grid.size() != D.size() || Grid.size() != W.size()
      || Grid.size() != E.size()) {
    return InvalidParameter;
  }

  const size_t gridsize = Grid.size();

  for (size_t i = 0; i < gridsize; i++) {
    double A = 0.0;
    RemezStatus status = ComputeA(Grid[i], ad, x, y, &A);

    if (status != Ok) {
      return status;
    }

    E[i] = W[i] * (D[i] - A);
  }

  return Ok;
}

/************************
 * Search
 *========
 * Searches for the maxima/minima of the error curve.  If more than
 * r+1 extrema are found, it uses the following heuristic (thanks
 * Chris Hanson):
 * 1) Adjacent non-alternating extrema deleted first.
 * 2) If there are more than one excess extrema, delete the
 *    one with the smallest error.  This will create a non-alternation
 *    condition that is fixed by 1).
 * 3) If there is exactly one excess extremum, delete the smaller
 *    of the first/last extremum
 *
 *
 * INPUT:
 * ------
 * size_t    r        - 1/2 the number of filter coefficients
 * size_t    Ext[]    - Indexes to Grid[] of extremal frequencies [r+1]
 * size_t    gridsize - Number of elements in the dense frequency grid
 * double E[]      - Array of error values.  [gridsize]
 * OUTPUT:
 * -------
 * size_t    Ext[]    - New indexes to extremal frequencies [r+1]
 ************************/
[[nodiscard]] static RemezStatus Search(
    vector<size_t>& Ext,
    vector<double>& E) {
  const size_t gridsize = E.size();
  const size_t r = Ext.size() - 1;
  // size_t up, alt;
  const size_t foundExtSize = gridsize; // originally 2r
  vector<size_t> foundExt(foundExtSize); /* Array of found extremals */


  /*
   * Allocate enough space for found extremals.
   */
  size_t k = 0;

  /*
   * Check for extremum at 0.
   */
  if (((E[0] > 0.0) && (E[0] > E[1])) || ((E[0] < 0.0) && (E[0] < E[1])))
    foundExt[k++] = 0;

  /*
   * Check for extrema inside dense grid
   */
  for (size_t i = 1; i < gridsize - 1; i++) {
    if (((E[i] >= E[i - 1]) && (E[i] > E[i + 1]) && (E[i] > 0.0))
        || ((E[i] <= E[i - 1]) && (E[i] < E[i + 1]) && (E[i] < 0.0))) {
      // PAK: we sometimes get too many extremal frequencies
      if (k >= foundExtSize) {
        return TooManyExtremalFrequencies;
      }
      foundExt[k++] = i;
    }
  }

  /*
   * Check for extremum at 0.5
   */
  const size_t lastGridIndex = gridsize - 1;
  if (((E[lastGridIndex] > 0.0) && (E[lastGridIndex] > E[lastGridIndex - 1]))
      || ((E[lastGridIndex] < 0.0)
          && (E[lastGridIndex] < E[lastGridIndex - 1]))) {
    if (k >= foundExtSize) {
      return TooManyExtremalFrequencies;
    }
    foundExt[k++] = lastGridIndex;
  }

  // PAK: we sometimes get not enough extremal frequencies
  if (k < r + 1) {
    fprintf(stderr, "Too few extremal frequencies: k (%zd) < r (%zd) + 1\n", k, r);
    return TooFewExtremalFrequencies;
  }

  /*
   * Remove extra extremals
   */
  size_t extra = k - (r + 1);
  //   assert(extra >= 0);

  while (extra > 0) {
    size_t up;

    if (E[foundExt[0]] > 0.0)
      up = 1; /* first one is a maxima */
    else
      up = 0; /* first one is a minima */

    size_t l = 0;
    size_t alt = 1;
    for (size_t j = 1; j < k; j++) {
      if (fabs(E[foundExt[j]]) < fabs(E[foundExt[l]]))
        l = j; /* new smallest error. */
      if ((up) && (E[foundExt[j]] < 0.0))
        up = 0; /* switch to a minima */
      else if ((!up) && (E[foundExt[j]] > 0.0))
        up = 1; /* switch to a maxima */
      else {
        alt = 0;
        // PAK: break now and you will delete the smallest overall
        // extremal.  If you want to delete the smallest of the
        // pair of non-alternating extremals, then you must do:
        //
        // if (fabs(E[foundExt[j]]) < fabs(E[foundExt[j-1]])) l=j;
        // else l=j-1;
        break; /* Ooops, found two non-alternating */
      }        /* extrema.  Delete smallest of them */
    }          /* if the loop finishes, all extrema are alternating */

    /*
     * If there's only one extremal and all are alternating,
     * delete the smallest of the first/last extremals.
     */
    if ((alt) && (extra == 1)) {
      if (fabs(E[foundExt[k - 1]]) < fabs(E[foundExt[0]]))
        /* Delete last extremal */
        l = k - 1;
      // PAK: changed from l = foundExt[k-1];
      else
        /* Delete first extremal */
        l = 0;
      // PAK: changed from l = foundExt[0];
    }

    for (size_t j = l; j < k - 1; j++) /* Loop that does the deletion */
    {
      foundExt[j] = foundExt[j + 1];
      //  assert(foundExt[j]<gridsize);
    }
    k--;
    extra--;
  }

  for (size_t i = 0; i <= r; i++) {
    //      assert(foundExt[i]<gridsize);
    Ext[i] = foundExt[i]; /* Copy found extremals to Ext[] */
  }

  return Ok;
}

/*********************
 * FreqSample
 *============
 * Simple frequency sampling algorithm to determine the impulse
 * response outTaps[] from A's found in ComputeA
 *
 *
 * INPUT:
 * ------
 * double   A[]      - Sample points of desired response [N/2]
 * Symmetry symmetry - Symmetry of desired filter
 *
 * OUTPUT:
 * -------
 * double outTaps[] - Impulse Response of final filter [N]
 *********************/
[[nodiscard]] static RemezStatus FreqSample(
    const vector<double>& A,
    vector<double>& outTaps,
    Symmetry symmetry) {
  const uint32_t N = A.size();

  if (N == 0) {
    return InvalidParameter;
  }

  double M = (N - 1.0) / 2.0;
  if (symmetry == POSITIVE) {
    if (N % 2) {
      for (uint32_t n = 0; n < N; n++) {
        double val = A[0];
        double x = Pi2 * (n - M) / N;
        for (uint32_t k = 1; k <= M; k++) {
          val += 2.0 * A[k] * cos(x * k);
        }
        outTaps[n] = val / N;
      }
    } else {
      for (uint32_t n = 0; n < N; n++) {
        double val = A[0];
        double x = Pi2 * (n - M) / N;
        for (uint32_t k = 1; k <= (N / 2 - 1); k++) {
          val += 2.0 * A[k] * cos(x * k);
        }
        outTaps[n] = val / N;
      }
    }
  } else {
    if (N % 2) {
      for (uint32_t n = 0; n < N; n++) {
        double val = 0;
        double x = Pi2 * (n - M) / N;
        for (uint32_t k = 1; k <= M; k++) {
          val += 2.0 * A[k] * sin(x * k);
        }
        outTaps[n] = val / N;
      }
    } else {
      for (uint32_t n = 0; n < N; n++) {
        double val = A[N / 2] * sin(Pi * (n - M));
        double x = Pi2 * (n - M) / N;
        for (uint32_t k = 1; k <= (N / 2 - 1); k++) {
          val += 2.0 * A[k] * sin(x * k);
        }
        outTaps[n] = val / N;
      }
    }
  }

  return Ok;
}

/*******************
 * isDone
 *========
 * Checks to see if the error function is small enough to consider
 * the result to have converged.
 *
 * INPUT:
 * ------
 * size_t    r     - 1/2 the number of filter coeffiecients
 * size_t    Ext[] - Indexes to extremal frequencies [r+1]
 * double E[]   - Error function on the dense grid [gridsize]
 *
 * OUTPUT:
 * -------
 * Returns 1 if the result converged
 * Returns 0 if the result has not converged
 ********************/

[[nodiscard]] static bool isDone(
    const vector<size_t>& Ext,
    const vector<double>& E) {
  const size_t r = Ext.size() - 1;
  double min, max, current;

  min = max = fabs(E[Ext[0]]);
  for (size_t i = 1; i <= r; i++) {
    current = fabs(E[Ext[i]]);
    if (current < min) min = current;
    if (current > max) max = current;
  }

  return (((max - min) / max) < 0.0001);
}

/********************
 * remez
 *=======
 * Calculates the optimal (in the Chebyshev/minimax sense)
 * FIR filter impulse response given a set of band edges,
 * the desired reponse on those bands, and the weight given to
 * the error in those bands.
 *
 * INPUT:
 * ------
 * size_t     numtaps     - Number of filter coefficients
 * size_t     numband     - Number of bands in filter specification
 * double  bands[]      - User-specified band edges [2 * numband]
 * double  des[]        - User-specified band responses [2 * numband]
 * double  weight[]     - User-specified error weights [numband]
 * FilterType type        - Type of filter
 * size_t     griddensity - ??
 *
 * OUTPUT:
 * -------
 * double h[]      - Impulse response of final filter [numtaps]
 ********************/
RemezStatus remezNew(
    vector<double>& outTaps,
    const vector<Band>& bands,
    FilterType type,
    size_t griddensity,
    size_t maxIterations) {
  if (outTaps.empty() || bands.empty() || griddensity <= 0
      || maxIterations <= 0) {
    return InvalidParameter;
  }

  Symmetry symmetry;

  if (type == Bandpass)
    symmetry = POSITIVE;
  else
    symmetry = NEGATIVE;

  size_t r = outTaps.size() / 2; /* number of extrema */
  if ((outTaps.size() % 2) && (symmetry == POSITIVE)) r++;
  outTaps[0] = 32;
  /*
   * Predict dense grid size in advance for memory allocation
   */
  size_t gridsize = 0;
  for (const auto& band : bands) {
    gridsize += lrint(
        static_cast<double>(2 * r * griddensity)
        * (band.highFrequency - band.lowFrequency));
  }
  if (symmetry == NEGATIVE) {
    gridsize--;
  }

  /*
   * Dynamically allocate memory for arrays with proper sizes
   */
  vector<double> Grid(gridsize);
  vector<double> D(gridsize);
  vector<double> W(gridsize);
  vector<double> E(gridsize);
  vector<size_t> Ext(r + 1);
  vector<double> taps(r + 1);
  vector<double> x(r + 1);
  vector<double> y(r + 1);
  vector<double> ad(r + 1);

  /*
   * Create dense frequency grid
   */
  RemezStatus status = CreateDenseGrid(
      r,
      outTaps.size(),
      bands,
      Grid,
      D,
      W,
      symmetry,
      griddensity);
  if (status != Ok) {
    return status;
  }

  status = InitialGuess(Ext, gridsize);
  if (status != Ok) {
    return status;
  }

  /*
   * For Differentiator: (fix grid)
   */
  if (type == Differentiator) {
    for (size_t i = 0; i < gridsize; i++) {
      /* D[i] = D[i]*Grid[i]; */
      if (D[i] > 0.0001) W[i] = W[i] / Grid[i];
    }
  }

  /*
   * For odd or Negative symmetry filters, alter the
   * D[] and W[] according to Parks McClellan
   */
  double c;
  if (symmetry == POSITIVE) {
    if (outTaps.size() % 2 == 0) {
      for (size_t i = 0; i < gridsize; i++) {
        c = cos(Pi * Grid[i]);
        D[i] /= c;
        W[i] *= c;
      }
    }
  } else {
    if (outTaps.size() % 2) {
      for (size_t i = 0; i < gridsize; i++) {
        c = sin(Pi2 * Grid[i]);
        D[i] /= c;
        W[i] *= c;
      }
    } else {
      for (size_t i = 0; i < gridsize; i++) {
        c = sin(Pi * Grid[i]);
        D[i] /= c;
        W[i] *= c;
      }
    }
  }

  /*
   * Perform the Remez Exchange algorithm
   */
  for (size_t iter = 0; iter < maxIterations; iter++) {
    status = CalcParms(Ext, Grid, D, W, ad, x, y);
    if (status != Ok) {
      return status;
    }

    status = CalcError(ad, x, y, Grid, D, W, E);
    if (status != Ok) {
      return status;
    }

    status = Search(Ext, E);
    if (status != Ok) {
      return status;
    }
    //      for(i=0; i <= r; i++) assert(Ext[i]<gridsize);
    if (isDone(Ext, E)) break;
  }

  if (!isDone(Ext, E)) {
    return DidNotConverge;
  }

  status = CalcParms(Ext, Grid, D, W, ad, x, y);
  if (status != Ok) {
    return status;
  }

  /*
   * Find the 'taps' of the filter for use with Frequency
   * Sampling.  If odd or Negative symmetry, fix the taps
   * according to Parks McClellan
   */
  for (size_t i = 0; i <= outTaps.size() / 2; i++) {
    if (symmetry == POSITIVE) {
      if (outTaps.size() % 2)
        c = 1;
      else
        c = cos(Pi * (double)i / (double)outTaps.size());
    } else {
      if (outTaps.size() % 2)
        c = sin(Pi2 * (double)i / (double)outTaps.size());
      else
        c = sin(Pi * (double)i / (double)outTaps.size());
    }
    double A = 0.0;
    status = ComputeA((double)i / (double)outTaps.size(), ad, x, y, &A);
    if (status != Ok) {
      return status;
    }

    taps[i] = A * c;
  }

  /*
   * Frequency sampling design with calculated taps
   */
  return FreqSample(taps, outTaps, symmetry);
}
