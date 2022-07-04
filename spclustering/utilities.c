#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "utilities.h"

void error(const char * message,...)
{
  va_list args;
  va_start(args,message);               /* Init 'args' to the beginning of */
                                        /* the variable length list of args*/
  fprintf(stderr,"\nerror:\n"); 
  vfprintf(stderr,message,args);
  fputs("\n",stderr);
  exit (1);
}

unsigned int* InitUIVector( long n ) {
   unsigned int* p;
   p = malloc( (size_t) (n*sizeof(unsigned int)) );
   assure( p, "allocation error" );
   return p;
}

int* InitIVector( long n ) {
   int* p;
   p = malloc( (size_t) (n*sizeof(int)) );
   assure( p, "allocation error" );
   return p;
}

char* InitCVector( long n ) {
   char* p;
   p = malloc( (size_t) (n*sizeof(char)) );
   assure( p, "allocation error" );
   return p;
}

float* InitVector( long n ) {
   float* p;
   p = malloc( (size_t) (n*sizeof(float)) );
   assure( p, "allocation error" );
   return p;
}

unsigned int** InitUIMatrix( long n, long j ) {
   unsigned int** m;
   int i;
   m = malloc( (size_t) (n*sizeof(unsigned int*)) );
   assure( m, "allocation error" );
   for( i=0; i<n; i++ ) {
      m[i] = malloc( (size_t) (j*sizeof(unsigned int)) );
      assure( m[i], "allocation error" );
   }
   return m;
}

void FreeUIMatrix( unsigned int** m, long n ) {
   int i;
   for( i=0; i<n; i++ ) free( m[i] );
   free( m );
}

int** InitIMatrix( long n, long j ) {
   int** m;
   int i;
   m = malloc( (size_t) (n*sizeof(int*)) );
   assure( m, "allocation error" );
   for( i=0; i<n; i++ ) {
      m[i] = malloc( (size_t) (j*sizeof(int)) );
      assure( m[i], "allocation error" );
   }
   return m;
}

void FreeIMatrix(int** m, long n ) {
   int i;
   for( i=0; i<n; i++ ) free( m[i] );
   free( m );
}

char** InitCMatrix( long n, long j ) {
   char** m;
   int i;
   m = malloc( (size_t) (n*sizeof(char*)) );
   assure( m, "allocation error" );
   for( i=0; i<n; i++ ) {
      m[i] = malloc( (size_t) (j*sizeof(char)) );
      assure( m[i], "allocation error" );
   }
   return m;
}

void ResetCMatrix( char** m, long n, long j ) {
   int i;
   for( i=0; i<n; i++ )
      memset( m[i], 0, j*sizeof(char) );
}

void FreeCMatrix( char** m, long n ) {
   int i;
   for( i=0; i<n; i++ ) free( m[i] );
   free( m );
}

double** InitDMatrix( long n, long j ) {
   double** m;
   int i;
   m = malloc( (size_t) (n*sizeof(double*)) );
   assure( m, "allocation error" );
   for( i=0; i<n; i++ ) {
      m[i] = malloc( (size_t) (j*sizeof(double)) );
      assure( m[i], "allocation error" );
   }
   return m;
}

void ResetDMatrix( double** m, long n, long j ) {
   int i;
   for( i=0; i<n; i++ )
      memset( m[i], 0, j*sizeof(double) );
}

void FreeDMatrix( double** m, long n ) {
   int i;
   for( i=0; i<n; i++ ) free( m[i] );
   free( m );
}
   
float** InitMatrix( long n, long j ) {
   float** m;
   int i;
   m = malloc( (size_t) (n*sizeof(float*)) );
   assure( m, "allocation error" );
   for( i=0; i<n; i++ ) {
      m[i] = malloc( (size_t) (j*sizeof(float)) );
      assure( m[i], "allocation error" );
   }
   return m;
}

void FreeMatrix( float** m, long n ) {
   int i;
   for( i=0; i<n; i++ ) free( m[i] );
   free( m );
}
   

/*                                                        */
/**********************************************************/





