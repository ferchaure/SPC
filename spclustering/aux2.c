#include "SW.h"

/**
   \section{AverageInteraction}
   \subsection{Description}
   calculates the average interaction between neighbors.
   \subsection{Input parameters}
   \begin{itemize}
   \item[J] the interaction array. J.p[i][j] is the interaction between point i
       and its jth neighbour.
   \end{itemize}
   \subsection{Return value}
   returns ``the Characteristic Distance''.
   \subsection{file}
   aux2.c
**/
float AverageInteraction( RaggedArray J )
{
int     i,k;
int     sum = 0;
float   J1 = 0;
   
   for(i = 0;  i < J.n; i++)
     for(k = 0; k<J.c[i]; k++) {
       J1 += J.p[i][k];
       sum++;
     }
   
   J1 /= ((float) sum);
   
   return(J1);
}

/**
   \section{GlobalCorrelation}
   \subsection{Description}
   Builds the correlations array. If two points belongs to the same cluster,
   one is added to the coresponding matrix element.
   \subsection{Input parameters}
   \begin{itemize}
   \item[CorrN] the correlations array. CorrN.p[i][j] is the number of times
       vertex i and vertex NK.p[i][j] were in the same cluster so far.
   \item[NK] nearest neighbours array. NK.p[i][j] is the j-th neighbour of
       vertex i.
   \item[Block] the cluster each vertex belongs to.
       vertex i belongs to cluster number Block[i].
   \end{itemize}
   \subsection{Output parameters}
   \begin{itemize}
   \item[CorrN] the updated correlation array.
   \end{itemize}
   \subsection{file}
   aux2.c
**/
void  GlobalCorrelation( UIRaggedArray CorrN, UIRaggedArray NK,
			 unsigned int *Block )
{
   int  i,k;

   for(i = 0; i < NK.n; i++)
      for(k = 0; k<NK.c[i]; k++)
	 if( Block[i] == Block[ NK.p[i][k] ] ) CorrN.p[i][k]++;
           
   return;
}

/**
   \section{FourPointCorrelation}
   \subsection{Description}
   Builds the four point correlations array. If four points belongs to
   the same cluster, one is added to the coresponding matrix element.
   \subsection{Input parameters}
   \begin{itemize}
   \item[FPCorr] the four point correlation array. CorrN.p[i][j].p[k][l]
   is the number of times vertices i, NK.p[i][j], k and NK.p[k][l]
   were in the same cluster so far.
   \item[NK] nearest neighbours array. NK.p[i][j] is the j-th neighbour of
       vertex i.
   \item[Block] the cluster each vertex belongs to.
       vertex i belongs to cluster number Block[i].
   \end{itemize}
   \subsection{Output parameters}
   \begin{itemize}
   \item[FPCorr] the updated four point correlation array.
   \end{itemize}
   \subsection{file}
   aux2.c
**/
void  FourPointCorrelation( RARaggedArray FPCorr, UIRaggedArray NK,
			    unsigned int *Block) {
  int i,k;
  int i1,k1;
  
  for(i = 0; i < NK.n; i++) 
    for(k = 0; k<NK.c[i]; k++) 
      if(NK.p[i][k]>i) 
        for(i1 = 0; i1 < NK.n; i1++)
          for(k1 = 0;  k1<NK.c[i1]; k1++) 
            if (NK.p[i1][k1]>i1) 
              if(Block[i] == Block[NK.p[i][k]] 
		 && Block[i1]==Block[NK.p[i1][k1]])
                (FPCorr.p[i][k]).p[i1][k1]++;
}



/* an auxiliary function for magnetization() and OrderClusterSize(). */
/* returns the opposite result to this of uicomp in edge.c           */
int uicompare(const void *i, const void *j)
{ return (int)( *((unsigned int*)j) - *((unsigned int*)i) ); }

/**
   \section{Magnetization}
   \subsection{Description}
   Obtain how many points have each spin value. order the groups in
   discending order of size. Calculate the magnetization for each spin
   color as \[ m_q = {Q N_q - N \over N (Q-1)}, \] where $N_q$ in the
   number of points with spin color $q$.
   \subsection{Input parameters}
   \begin{itemize}
   \item[N] number of points.
   \item[Q] number of spin values (colors).
   \item[nc] number of clusters.
   \item[*ClusterSize] cluster sizes.
   \item[$N_q$] previously allocated workspace of 
       size $>=$ Q*sizeof(unsigned int).
   \end{itemize}
   \subsection{Output parameters}
   \begin{itemize}
   \item[mag] the magnetization vector.
   \end{itemize}
   \subsection{Auxiliary function}
   int uicompare(const void *i, const void *j)
   \subsection{file}
   aux2.c
**/
float Magnetization( int N, int Q, int nc, unsigned int *ClusterSize,
		     float* mag, unsigned int *N_q )
{
   int k, q;

   memset( N_q, 0, Q*sizeof(unsigned int) );
   for(k = 0; k < nc; k++)
      N_q[ IRAND(Q) ] += ClusterSize[k];
   qsort(N_q,Q,sizeof(unsigned int),uicompare);
   for(q = 0; q < Q; q++)
      mag[q] = (float)( (int)(Q * N_q[q] - N) ) / (N*(Q - 1.0));

   return mag[0];
}

/**
   \section{OrderClusterSize}
   \subsection{Description}
   Order cluster sizes list in discending order..
   \subsection{Input parameters}
   \begin{itemize}
   \item[nc] number of clusters.
   \item[*ClusterSize] cluster sizes.
   \end{itemize}
   \subsection{Output parameters}
   \begin{itemize}
   \item[*ClusterSize] ordered cluster sizes.
   \end{itemize}
   \subsection{Auxiliary function}
   int uicompare(const void *i, const void *j)
   \subsection{file}
   aux2.c
**/
void OrderClusterSize( int nc, unsigned int *ClusterSize )
{
   qsort(ClusterSize,nc,sizeof(unsigned int),uicompare);   
}   

/**
   \section{ClusterAverage}
   \subsection{Description}
   Calculate cluster size averages. Returns the mean value and
   variance of the larger, second larger, .., smaller cluster sizes.
   \subsection{Input parameters}
   \begin{itemize}
   \item[nc] number of clusters.
   \item[*Size1] cluster size conumlant. The i-th element is the comulant of
          the i-th larger cluster size.
   \item[*size2] cluster size$^2$ comulant.
   \end{itemize}
   \subsection{Output parameters}
   \begin{itemize}
   \item[*Size1] sizes mean value.
   \item[*Size2] sizes variance.
   \end{itemize}
   \subsection{file}
   aux2.c
**/
void ClusterAverage(int ncy, int N, float *Size1, float *Size2)
{
   int i;
   /* assume Size1[] is in descending order */
   for(i = 0; i<N && Size1[i]>0; i++) {
      Size1[i] /= (float)ncy;
      Size2[i] /= (float)ncy;
      Size2[i] -= (Size1[i] * Size1[i]);
   }
}

/**
   \section{Susceptibility}
   \subsection{Description}
   Calculate magnatizations mean value and susceptibilities.
   \subsection{Input parameters}
   \begin{itemize}
   \item[Q] number of spin values.
   \item[ncy] number of SW sweeps performed = numbers of samples taken.
   \item[*M1] magnetizations comulant.
   \item[*M2] magnetizations$^2$ comulant.
   \end{itemize}
   \subsection{Output parameters}
   \begin{itemize}
   \item[*M1] magnetizations mean value.
   \item[*xi] susceptibilities.
   \end{itemize}
   \subsection{file}
   aux2.c
**/
void Susceptibility( int Q, int ncy, float* M1, float* M2, float* xi )
{
   int q;
   for(q=0;q<Q;q++) {
      M1[q] /= (float)ncy;
      M2[q] /= (float)ncy;          
      xi[q] = (M2[q] -  M1[q] * M1[q]); 
   }
}

int  Thresholding(int ncy, float threshold,   
                  UIRaggedArray Corr, UIRaggedArray NK, CRaggedArray Bond,
                  unsigned int *Block, unsigned int *ClusterSize,
                  unsigned int *OldBlock, int *n_cols, unsigned int *ws)
{
   int i,k,nc;
   float  th;

   th = threshold * ncy;
     
   for(i=0; i< NK.n; i++)
       for(k = 0; k<NK.c[i];  k++)
         if(th < Corr.p[i][k]) {
           if ( OldBlock[i] == OldBlock[ NK.p[i][k] ] ) {
             Bond.p[i][k] = 1;
           }
           else {
             Bond.p[i][k] = 0;
             (*n_cols)++;
           }
         }
         else
	   Bond.p[i][k] = 0;
     
   nc = Coarsening(Bond,Block,NK,ClusterSize,ws);
   OrderingClusters(NK.n,nc,Block,ClusterSize,ws);
   
   return(nc);
}  

/* ---------------------------------------------------------------------- */
/*   We perform a directed growth in order to  determin  a partition      */
/*   Two points belongs to the same cluster if they have a common         */
/*   predesessor (see Fukunaga).                                          */
/*   The predessessor of each point is the point with maximal correlation */
/*   among its neighboors.                                                */
/*   The threshold is the number for which we consider that the           */
/*   correlation is "zero"                                                */
/* ---------------------------------------------------------------------- */
/*   actually, here each point is joined to the cluster of                */
/*   its predessessor   -- Guy */

int  DirectedGrowth( int ncy, float threshold,   
		     UIRaggedArray Corr, UIRaggedArray NK, UIRaggedArray KN,
		     CRaggedArray Bond, unsigned int *Block, 
		     unsigned int *ClusterSize, unsigned int *dgOldBlock,
		     unsigned int *thOldBlock, unsigned int *ws )
{
   int i,k;
   float th;
   float dh;
   int   nc;
   int   max, kmax;

   th = threshold * ncy;
   dh = th/20;

   for(i=0; i< NK.n; i++)
      for(k = 0; k<NK.c[i];k++)
         if ((th < Corr.p[i][k]) && 
	     (thOldBlock[i] == thOldBlock[NK.p[i][k]])) {
         Bond.p[i][k] = 1;
       }
       else Bond.p[i][k] = 0;

   for(i=0; i< NK.n; i++){
      max = 0;
      for(k = 0; k<NK.c[i];  k++) 
	 if( max < Corr.p[i][k] ){
	    max = Corr.p[i][k];
	    kmax = k;
	 }
      if(max > dh){
         if ( dgOldBlock[i] == dgOldBlock[ NK.p[i][kmax] ] ) {
	   Bond.p[i][kmax] = 1;
	   Bond.p[ NK.p[i][kmax] ][ KN.p[i][kmax] ] = 1;
	 }
      }
   }

   nc = Coarsening(Bond,Block,NK,ClusterSize,ws);
   OrderingClusters(NK.n,nc,Block,ClusterSize,ws);
   
   return(nc);
}    








