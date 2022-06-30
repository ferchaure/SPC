#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "param.h"
#include "utilities.h"

typedef struct{
  char name[PARAM_MAX_FIELD+1], val[PARAM_MAX_FIELD+1];
  int i;
  float f;
  void* next;
} parameter;

#define PLIST_SIZE 128

static parameter *plist[PLIST_SIZE];
static int init=0;

/**
\section{Use of Parameter environment}
   This environment allows you to set parameters, so that their values can 
   be read at any time and anywhere in the program. The parameters can be
   written and read as a char string, an integer, or a floating point
   argument. The following functions are available:
\begin{description}
**/

int hfunc( char* n ) {
   int h;
   h = 4 * (n[0] % 32) + n[1] % 4;
   assure( h<PLIST_SIZE, "h too large" );
   return h;
}

void init_plist() {
   memset( plist, 0, PLIST_SIZE*sizeof(parameter*) );
   init = 1;
}

parameter* find_param( char* n, int* r ) {
   int h;
   parameter *p, *pl;

   if( !init ) init_plist();

   h = hfunc( n );
   if( plist[h] == NULL ) {
      *r = h;
      return NULL;
   }
   for( p = plist[h]; p != NULL; p = p->next ) {
      if( strcmp( n, p->name ) == 0 ) {
	 *r = -1;
	 return p;
      }
      pl = p;
   }
   *r = -2;
   return pl;
}
   
/**
\item[int SetParam( char* n, char* v )] $\:$ \\
   Parameter n is set to string v. If v=NULL the parameter is set to
   a zero length string. Return 1 if parameter was already set,
   and 0 otherwise.
**/
int SetParam( char* n, char* v ) {
   parameter *p, *pf;
   int r, j;
   float l;
   char nul=0;
   if( v == NULL ) v = &nul;
   assure( strlen(n)<PARAM_MAX_FIELD && strlen(v)<PARAM_MAX_FIELD, "too long" );

   pf = find_param( n, &r );

   if( r != -1 ) {
      p = malloc( sizeof(parameter) );
      if( r >= 0 ) plist[r] = p;
      else pf->next = p;
      strcpy( p->name, n );
      p->next = NULL;
   }      
   else p=pf;

   strcpy( p->val, v );
   if( sscanf( v, "%d", &j ) != 1 ) p->i = 0;
   else p->i = j;
   if( sscanf( v, "%f", &l ) != 1 ) p->f = 0.;
   else p->f = l;

   return ( r == -1 )? 1 : 0;
}

/**
\item[int UnsetParam( char* n )] $\:$ \\
   Unset (delete) the parameter n.
   returns 1 if parameter existed, 0 otherwise.
**/
int UnsetParam( char* n ) {
   int h;
   parameter *p, *pl;

   if( !init ) init_plist();

   h = hfunc( n );
   pl = NULL;
   for( p = plist[h]; p != NULL; p = p->next ) {
      if( strcmp( n, p->name ) == 0 ) {
	 if( pl == NULL ) plist[h] = p->next;
	 else pl->next = p->next;
	 free( p );
	 return 1;
      }
      pl = p;
   }
   return 0;
}

/**
\item[char* GetParam( char* n )] $\:$ \\
   Returns a pointer to the string value of parameter n. If the parameter
   was set with NULL pointer, the function returns a pointer to zero length
   string. If the parameter is not set, the function returns NULL.
**/
char* GetParam( char* n ) {
   int r;
   parameter *pf;
   pf = find_param( n, &r );
   if( r != -1 )
     return NULL;
   return pf->val;
}

/**
\item[int IGetParam( char* n )] $\:$ \\
   Return the value of parameter n as an integer. If the parameter does not
   describe an integer or is not set, the function returns 0.
**/
int IGetParam( char* n ) {
   int r;
   parameter* pf;
   pf = find_param( n, &r );
   if( r != -1 )
     return 0;
   return pf->i;
}

/**
\item[int FGetParam( char* n )] $\:$ \\
   Return the value of parameter n as a float. If the parameter does not
   describe a float or is not set, the function returns 0.
**/
float FGetParam( char* n ) {
   int r;
   parameter* pf;
   pf = find_param( n, &r );
   if( r != -1 )
     return 0.;
   return pf->f;
}

void fprint_param( FILE* fp ) {
   int i;
   parameter *p;

   if( !init ) init_plist();

   for( i=0; i<PLIST_SIZE; i++ )
      for( p = plist[i]; p != NULL; p = p->next )
	 fprintf( fp, "%s\t%s\n", p->name, p->val );
}


/**
\item[int FSetParam( char* n, float l )] $\:$ \\
   Sets the value of parameter n to represent the floating point argument l.
   Returns 1 if parameter was already set, 0 otherwise.
**/
int FSetParam( char* n, float l ) {
   char c[PARAM_MAX_FIELD+1];
   sprintf( c, "%f", l );
   return SetParam( n, c );
}

/**
\item[int ISetParam( char* n, int j )] $\:$ \\
   Sets the value of parameter n to represent the integer j.
   Returns 1 if parameter was already set, 0 otherwise.
**/
int ISetParam( char* n, int j ) {
   char c[PARAM_MAX_FIELD+1];
   sprintf( c, "%d", j );
   return SetParam( n, c );
}

/**
\end{description}
**/








