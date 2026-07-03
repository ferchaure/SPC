#include "RaggedArray.h"
#include "utilities.h"
#include <stdlib.h>

UIRaggedArray InitUIRaggedArray(UIRaggedArray K) {
  UIRaggedArray M;
  int i;

  M.n = K.n;
  M.p = (unsigned int **)calloc(M.n, sizeof(unsigned int *));
  M.c = (unsigned int *)calloc(M.n, sizeof(unsigned int));
  assure(M.p && M.c, "allocation failure");
  for (i = 0; i < M.n; i++) {
    M.c[i] = K.c[i];
    if (M.c[i] != 0) {
      M.p[i] = (unsigned int *)calloc(M.c[i], sizeof(unsigned int));
      assure(M.p[i], "allocation failure");
    }
  }
  return M;
}

RaggedArray InitRaggedArray(UIRaggedArray K) {
  RaggedArray M;
  int i;

  M.n = K.n;
  M.p = (float **)calloc(M.n, sizeof(float *));
  M.c = (unsigned int *)calloc(M.n, sizeof(unsigned int));
  assure(M.p && M.c, "allocation failure");
  for (i = 0; i < M.n; i++) {
    M.c[i] = K.c[i];
    if (M.c[i] != 0) {
      M.p[i] = (float *)calloc(M.c[i], sizeof(float));
      assure(M.p[i], "allocation failure");
    }
  }
  return M;
}

CRaggedArray InitCRaggedArray(UIRaggedArray K) {
  CRaggedArray M;
  int i;

  M.n = K.n;
  M.p = (char **)calloc(M.n, sizeof(char *));
  M.c = (unsigned int *)calloc(M.n, sizeof(unsigned int));
  assure(M.p && M.c, "allocation failure");
  for (i = 0; i < M.n; i++) {
    M.c[i] = K.c[i];
    if (M.c[i] != 0) {
      M.p[i] = (char *)calloc(M.c[i], sizeof(char));
      assure(M.p[i], "allocation failure");
    }
  }
  return M;
}

void ResetUIRaggedArray(UIRaggedArray K) {
  int i;
  for (i = 0; i < K.n; i++)
    memset(K.p[i], 0, K.c[i] * sizeof(unsigned int));
}

void ResetCRaggedArray(CRaggedArray K) {
  int i;
  for (i = 0; i < K.n; i++)
    memset(K.p[i], 0, K.c[i] * sizeof(char));
}

void ResetRaggedArray(RaggedArray K) {
  int i;
  for (i = 0; i < K.n; i++)
    memset(K.p[i], 0, K.c[i] * sizeof(float));
}

void FreeUIRaggedArray(UIRaggedArray K) {
  int i;
  for (i = 0; i < K.n; i++)
    free(K.p[i]);
  free(K.p);
  free(K.c);
}

void FreeRaggedArray(RaggedArray K) {
  int i;
  for (i = 0; i < K.n; i++)
    free(K.p[i]);
  free(K.p);
  free(K.c);
}

void FreeCRaggedArray(CRaggedArray K) {
  int i;
  for (i = 0; i < K.n; i++)
    free(K.p[i]);
  free(K.p);
  free(K.c);
}

RARaggedArray InitRARaggedArray(UIRaggedArray K) {
  RARaggedArray M;
  int i, j;

  M.n = K.n;
  M.p = (UIRaggedArray **)calloc(M.n, sizeof(UIRaggedArray *));
  M.c = (unsigned int *)calloc(M.n, sizeof(unsigned int));
  assure(M.p && M.c, "allocation failure");
  for (i = 0; i < M.n; i++) {
    M.c[i] = K.c[i];
    if (M.c[i] != 0) {
      M.p[i] = (UIRaggedArray *)calloc(M.c[i], sizeof(UIRaggedArray));
      assure(M.p[i], "allocation failure");
    }
  }

  for (i = 0; i < M.n; i++)
    for (j = 0; j < M.c[i]; j++)
      M.p[i][j] = InitUIRaggedArray(K);

  return M;
}

void ResetRARaggedArray(RARaggedArray K) {
  int i, j;
  for (i = 0; i < K.n; i++)
    for (j = 0; j < K.c[i]; j++)
      ResetUIRaggedArray(K.p[i][j]);
}

void FreeRARaggedArray(RARaggedArray K) {
  int i, j;
  for (i = 0; i < K.n; i++) {
    for (j = 0; j < K.c[i]; j++)
      FreeUIRaggedArray(K.p[i][j]);
    free(K.p[i]);
  }
  free(K.p);
  free(K.c);
}
