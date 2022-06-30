#include "SW.h"

/**
   \section{InitialSpinConfig}
   \subsection{Description}
   Sets inital potts spins configuration.
   \subsection{Input parameters}
   \begin{itemize}
   \item[N] Number of points.
   \item[Q] Number of Potts spin values.
   \end{itemize}
   \subsection{Output parameters}
   \begin{itemize}
   \item[Spin] Spin[i] is the spin value ascociated with vertex i.
   \end{itemize}
   \subsection{file}
   aux1.c
**/
void  InitialSpinConfig(int N, unsigned int *Spin, int Q)
{
   int i;
   if ( GetParam("RandomInitialConfig") )
     for(i = 0; i < N; i++) Spin[i] = IRAND(Q);
   else
     memset( Spin, 0, N*sizeof(unsigned int) );

   return;
}

/**
   \section{NewSpinConfig}
   \subsection{Description}
   assign each cluster a random spin value.
   \subsection{Input parameters}
   \begin{itemize}
   \item[N] Number of points.
   \item[Block] point i belongs to cluster Block[i].
   \item[MBlk] number of clusters.
   \item[Q] Number of Potts spin values.
   \item[NewSpinValue] Previously allocated workspace. \\
      Must have minimal size of N*sizeof(unsigned int).
   \end{itemize}
   \subsection{Output parameters}
   \begin{itemize}
   \item[Spin] Spin[i] is the new spin value ascociated with vertex i.
   \end{itemize}
   \subsection{file}
   aux1.c
**/
void NewSpinConfig(          
   int            N,           
   unsigned int  *Spin,  
   unsigned int *Block, 
   int            NBlk,        
   int            Q,
   unsigned int *NewSpinValue ) 
{
   int  nb, i;

   for (nb = 0; nb < NBlk; nb++)
      NewSpinValue[nb] = IRAND(Q);
   for(i = 0; i < N; i++) Spin[i] = NewSpinValue[ Block[i] ];
}
 

/**
   \section{DeletionProbabilities}
   \subsection{Description}
   Gives the deletion probabilities for a satisfied bond. 
   If bond is unsatisfied its deletion probability is 1.
   \subsection{Input parameters}
   \begin{itemize}
   \item[T] Temperature.
   \item[J] J[i][j] is the interaction between i and NK.p[i][j],
        where NK is the edges array.
   \end{itemize}
   \subsection{Output parameters}
   \begin{itemize}
   \item[P] P.p[i][j] is the deletion probability of a satisfied bond 
       between vertices i and NK.p[i][j] when their spin values are equal.
   \end{itemize}
   \subsection{file}
   aux1.c
**/
void DeletionProbabilities( float T, RaggedArray J, RaggedArray P )
{
   int i,k;
   if( T == 0.0 ) ResetRaggedArray( P );
   else {
      for(i = 0; i < J.n; i++) 
	 for(k = 0; k<J.c[i]; k++){
	    if(J.p[i][k] != 0.0) P.p[i][k] = exp( - J.p[i][k] / T );
	    else P.p[i][k] = 1.0;
	 }
   }
}

/**
   \section{SetBond}
   \subsection{Description}
   Decides which bonds to freeze. 
   \subsection{Input parameters}
   \begin{itemize}
   \item[P] P.p[i][j] is the deletion probability of the edge
       between vertices i and NK.p[i][j] if they have the same spin.
   \item[Spin] Spin[i] is the current spin of vertex i.
   \item[NK] nearest neighbours array. For each vertex the list of
       neighbours \emph{must} be in ascending order.
   \item[KN] the inverse nearest neighbours array.
   \end{itemize}
   \subsection{Output parameters}
   \begin{itemize}
   \item[Bond] Frozen bond array. Bond.p[i][j] = 1 $\longrightarrow$ 
       bond between i and NK.p[i][j] is frozen.
       Bond.p[i][j] = 0 $\longrightarrow$ the bond is deleted.
   \end{itemize}
   \subsection{Return value}
   \begin{itemize}
   \item[nb] the number of frozen bonds.
   \end{itemize}
   \subsection{file}
   aux1.c
**/
int SetBond( RaggedArray P, unsigned int *Spin, CRaggedArray Bond,
             UIRaggedArray NK, UIRaggedArray KN){
   int  nb = 0; 
   int  i,k;

   for(i = 0; i < Bond.n; i++)
      for( k = Bond.c[i]-1; NK.p[i][k]>i && k>=0; k-- ) {
	 if( (Spin[i] == Spin[NK.p[i][k]] ) && (RAND(1.) > P.p[i][k]) ) {
	    Bond.p[i][k] = 1;
	    Bond.p[ NK.p[i][k] ][ KN.p[i][k] ] = 1;  
	    nb ++;
	 }
	 else{
	    Bond.p[i][k] = 0;
	    Bond.p[ NK.p[i][k] ][ KN.p[i][k] ] = 0;  
	 }
      }
   return nb;
}

/**
   \section{Coarsening}
   \subsection{Description}
   Uses the frozen bonds to indentify the connected components (clusters).
   Assign each vertex its cluster number.
   \subsection{Input parameters}
   \begin{itemize}
   \item[Bond] Frozen bonds array. Bond.p[i][j] = 1 $\longrightarrow$
       the bond between i and NK.p[i][j] is frozen.
       Bond.p[i][j] = 0 $\longrightarrow$ the bond is deleted.
   \item[NK] nearest neighbours array.
   \item[Stack] previously allocated workspace of 
       size $>=$ N*sizeof(unsigned int).
   \end{itemize}
   \subsection{Output parameters}
   \begin{itemize}
   \item[Block] assignment to clusters. Spin i belongs to cluster Block[i].
   \item[ClusterSize] cluster sizes. ClusterSize[k] is th size of cluster k.
   \end{itemize}
   \subsection{Return value}
   the number of clusters.
   \subsection{file}
   aux1.c
**/
int  Coarsening(CRaggedArray Bond, unsigned int *Block,
                UIRaggedArray NK, unsigned int *ClusterSize,
		unsigned int* Stack )
{
   int  ns,i,k,i0,N;
   int  nblock = -1;    /* number of clusters (blocks) generated */

   N = Bond.n;
   for(i = 0; i < N; i++) Block[i] = UINT_MAX;
   memset( ClusterSize, 0, N*sizeof(unsigned int) );

   for(i = 0; i < N; i++)
      if (Block[i] == UINT_MAX){        /* point i was not labeled yet */
        ns = 0;
        nblock ++;
        Block[i] = nblock;
        ClusterSize[nblock] = 1;
        Stack[ns] = i;
        while(ns >= 0) {
          i0 = Stack[ns];
          ns--;
          for(k = 0; k<Bond.c[i0]; k++){
            if( (Bond.p[i0][k] == 1) && (Block[ NK.p[i0][k] ] == UINT_MAX) ) {
              Block[ NK.p[i0][k] ] = nblock;
              ClusterSize[nblock] ++;
              ns++;
              Stack[ns] = NK.p[i0][k];
            }
          }
        }
      }

   nblock++;
   return(nblock);
}

/**
   \section{OrderingClusters}
   \subsection{Description}
   reassign cluster numbers according to cluster sizes, so if
   i $<$ j $\longrightarrow$ cluster i is larger than cluster j.
   \subsection{Input parameters}
   \begin{itemize}
   \item[N] number of vertices.
   \item[nblock] number of clusters.
   \item[Block] vertex i belongs to cluster Block[i].
   \item[Size] Cluster sizes. Cluster i contains Size[i] vertices.
   \item[Indx] previously allocated workspace of 
       size $>=$ 2*N*sizeof(unsigned int).
   \end{itemize}
   \subsection{Output parameters}
   \begin{itemize}
   \item[Block] new cluster number assigned to each vertex.
   \item[Size] cluster sizes, ordered in descending order, according
        to their new numbers.
   \end{itemize}
   \subsection{file}
   aux1.c
**/
void  OrderingClusters(     
  int            N,          
  int            nblock,     
  unsigned int  *Block,
  unsigned int  *Size,
  unsigned int  *Indx )
{
   int i;
   unsigned int *NewIndx;
   
   if( nblock == 1 )
      return;

   NewIndx = Indx+N;
   memset( Indx, 0, N*sizeof(unsigned int) );

   /* Size[i] is the size of block i */

   /* sort the indexes (linear in N) so if Size[i]>Size[j] -> Indx[i]<Indx[j] */
   for( i=0; i<nblock; i++ )
      Indx[ Size[i] ] ++;
   NewIndx[N-1] = 0;
   for( i=N-2; i>0; i-- )
      NewIndx[i] = NewIndx[i+1] + Indx[i+1];
   for( i=0; i<nblock; i++ ) {
      Indx[i] = NewIndx[ Size[i] ];
      NewIndx[ Size[i] ] ++;
   }

   /* order sizes according to indexes Size[i]--->Size[Indx[i]] */
   for( i=0; i<nblock; i++ )
      NewIndx[ Indx[i] ] = Size[ i ];
   memcpy( Size, NewIndx, nblock*sizeof(unsigned int) );
   
   
   /* now i<j  ->  Size[i]>=Size[j]   */
   /* and Size[Indx[i]] is the size of block i. */

   /* Asign to block number Indx[i] the new number i */
   for( i=0; i<N; i++ ) Block[i] = Indx[ Block[i] ];

   return;
}

/**
   \section{CheckParam}
   \subsection{Description}
   Check evironment parameters in order to prevent runtime error.
   \subsection{file}
   aux1.c
**/
void CheckParam()
{
   int Q,N;
   assure( (N=IGetParam("NumberOfPoints")) > 0 , "N<=0" );
   assure( IGetParam("Dimensions") >= 0 , "D<0" );
   assure( (Q=IGetParam("PottsSpins")) <= UINT_MAX , "Q too large" );
   assure( Q>1, "Q too small" );
   assure( IGetParam("SWCycles") < UINT_MAX , "cyc too large" );
   /* if cyc too large can change unsigned int Corr -> unsigned long Corr */
   assure( N<UINT_MAX, "N too large" );
   assure( FGetParam( "ThresholdStep" )>0.0 || 
	   FGetParam( "ThresholdTheta" )>0.0, "theta is zero" );
   assure( IGetParam( "ClustersReported" )<=N, "NS > N" );
}

/**
   \section{CheckParam}
   \subsection{Description}
   Set default values to environmrnt parameters.
   \subsection{file}
   aux1.c
**/
void DefaultParam()
{
   ISetParam( "PottsSpins",     20         );
   ISetParam( "SWCycles",       2500       );
   FSetParam( "SWFraction",     0.8        );
   FSetParam( "ThresholdTheta", 0.5        );
   SetParam(  "OutFile",        "SWout"    );
   ISetParam( "ClustersReported", 12       );
   ISetParam( "SusceptColors",    4        );
}

/**
   \section{Energy}
   \subsection{Description}
   Calculate the energy for a given configuration:
   \[ E(\{s_i\})=\sum_{i,j} J_{ij}(1-\delta_{s_i s_j}) \]
   \subsection{Input parameters}
   \begin{itemize}
   \item[Block] vertex i belongs to cluster Block[i].
   \item[J] Interactions array.
   \item[NK] Neighbours array.
   \end{itemize}
   \subsection{Return value}
   the energy
   \subsection{file}
   aux1.c
**/
double Energy( unsigned int* Block, RaggedArray J, UIRaggedArray NK  )
{
   int i, k;
   double e=0;
   for(i = 0; i < J.n; i++)
      for( k = J.c[i]-1; NK.p[i][k]>i && k>=0; k-- )
	 if( Block[i] != Block[NK.p[i][k]] ) 
	    e += J.p[i][k];
   return e;
}



