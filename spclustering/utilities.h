#ifndef UTILITIES_H
#define UTILITIES_H

#define INF INFINITY
// #ifdef INFINITY
// #define INF INFINITY
// #endif
#include <string.h>

#define assure(expr,message)                            \
        if      (expr) ;                                \
        else error("at line %d of '%s': %s",__LINE__,__FILE__,message);
#define RAND(i)   ( (double)(i)*rand() / ((double)RAND_MAX+.01) )
#define IRAND(i)  ( (int)RAND(i) )

void error( const char * message, ... );
unsigned int* InitUIVector( long n );
int* InitIVector( long n );
char* InitCVector( long n );
float* InitVector( long n );
double** InitDMatrix( long n, long m );
float** InitMatrix( long n, long j );
void FreeDMatrix( double** m, long n );
unsigned int** InitUIMatrix( long n, long m );
void FreeUIMatrix( unsigned int** m, long n );
int** InitIMatrix( long n, long m );
void FreeIMatrix( int** m, long n );
char** InitCMatrix( long n, long m );
void FreeCMatrix( char** m, long n );
void FreeMatrix( float** m, long n );
void ResetCMatrix( char** m, long n, long j );
void ResetDMatrix( double** m, long n, long j );

#endif
