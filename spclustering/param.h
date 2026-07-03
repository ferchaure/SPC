#ifndef PARAM_H
#define PARAM_H

#include <stdio.h>

#define PARAM_MAX_FIELD 50

int SetParam(char *n, char *v);
int UnsetParam(char *n);
char *GetParam(char *n);
int IGetParam(char *n);
float FGetParam(char *n);
void fprint_param(FILE *fp);
int FSetParam(char *n, float l);
int ISetParam(char *n, int j);

#endif
