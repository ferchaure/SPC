#ifndef RAGGEDARRAY_H
#define RAGGEDARRAY_H

typedef struct {
  float **p;
  unsigned int *c;
  unsigned int n;
} RaggedArray;

typedef struct {
  unsigned int **p;
  unsigned int *c;
  unsigned int n;
} UIRaggedArray;

typedef struct {
  char **p;
  unsigned int *c;
  unsigned int n;
} CRaggedArray;

typedef struct {
  UIRaggedArray **p;
  unsigned int *c;
  unsigned int n;
} RARaggedArray;

RaggedArray InitRaggedArray(UIRaggedArray K);
UIRaggedArray InitUIRaggedArray(UIRaggedArray K);
CRaggedArray InitCRaggedArray(UIRaggedArray K);
void ResetUIRaggedArray(UIRaggedArray K);
void ResetCRaggedArray(CRaggedArray K);
void ResetRaggedArray(RaggedArray K);
void FreeUIRaggedArray(UIRaggedArray K);
void FreeRaggedArray(RaggedArray K);
void FreeCRaggedArray(CRaggedArray K);

RARaggedArray InitRARaggedArray(UIRaggedArray K);
void ResetRARaggedArray(RARaggedArray K);
void FreeRARaggedArray(RARaggedArray K);

#endif
