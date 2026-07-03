#include "SW.h"

/* -------------------------------------------------------------------- */
/*  Distance: returns the euclidean distance between data poins i and j */
/* -------------------------------------------------------------------- */
double Distance(int D, double *X, double *Y) {
  int d;
  double dist = 0.0;
  double diff;
  for (d = 0; d < D; d++) {
    diff = X[d] - Y[d];
    dist += diff * diff;
  }
  return (sqrt(dist));
}
/* -------------------------------------------------------------------- */
/*  Distance: returns the squared  euclidean distance between data poins i and j
 */
/* -------------------------------------------------------------------- */
double Squared_Distance(int D, double *X, double *Y) {
  int d;
  double dist = 0.0;
  double diff;
  for (d = 0; d < D; d++) {
    diff = X[d] - Y[d];
    dist += diff * diff;
  }
  return (dist);
}

/* -------------------------------------------------------------------- */
/*  Distance_Linf: returns the L-infinity distance between data poins i */
/*                 and j                                                */
/* -------------------------------------------------------------------- */
double Distance_Linf(int D, double *X, double *Y) {
  int d;
  double dist;
  double diff;
  dist = 0.0;
  for (d = 0; d < D; d++) {
    diff = fabs(X[d] - Y[d]);
    if (dist < diff)
      dist = diff;
  }
  return (dist);
}

RaggedArray EdgeDistance(int D, UIRaggedArray NK, double **X) {
  RaggedArray M;
  int i, k, metric, nedges;
  float chd;

  M = InitRaggedArray(NK);
  metric = (GetParam("InfMetric") == NULL);

  chd = 0.0;
  nedges = 0;
  for (i = 0; i < M.n; i++) {
    for (k = 0; k < M.c[i]; k++) {
      M.p[i][k] = D ? (metric ? Distance(D, X[i], X[NK.p[i][k]])
                              : Distance_Linf(D, X[i], X[NK.p[i][k]]))
                    : X[i][NK.p[i][k]];
      if (M.p[i][k] < INF)
        chd += M.p[i][k], nedges++;
    }
  }
  nedges /= 2;

  ISetParam("NumberOfEdges", nedges);
  FSetParam("NearestNeighbrs", 2.0 * (float)nedges / (float)M.n);
  FSetParam("CharDist", chd / (2.0 * (float)nedges));
  return M;
}

void DistanceToInteraction(RaggedArray M, UIRaggedArray NK, UIRaggedArray KN) {
  int i, k;
  float dd, lambdaq, *z, nn, chd;

  if ((nn = FGetParam("ForceNN")) == 0.)
    nn = FGetParam("NearestNeighbrs");
  if ((chd = FGetParam("ForceChD")) == 0.)
    chd = FGetParam("CharDist");

  /* the interactions are calculated */
  for (i = 0; i < M.n; i++)
    for (k = M.c[i] - 1; NK.p[i][k] > i && k >= 0; k--) {
      dd = (M.p[i][k] * M.p[NK.p[i][k]][KN.p[i][k]]) / (chd * chd);
      M.p[i][k] = M.p[NK.p[i][k]][KN.p[i][k]] = exp(-dd / 2.0) / nn;
    }

  /* calc z_i */
  if (GetParam("UseZ")) {
    /* allocate z */
    z = InitVector(M.n);
    for (i = 0; i < M.n; i++)
      for (k = 0; k < M.c[i]; k++) {
        z[i] += M.p[i][k];
      }
    lambdaq = FGetParam("Lambda") * (float)IGetParam("PottsSpins");
    for (i = 0; i < M.n; i++)
      for (k = 0; k < M.c[i]; k++) {
        M.p[i][k] *= (1.0 / (z[i] + lambdaq) + 1.0 / (z[NK.p[i][k]] + lambdaq));
      }
    free(z);
  }
}
