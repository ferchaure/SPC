#include "utilities.h"
#include "SW.h"


/**
   \section{ReadEdgeFile}
   \subsection{Description}
   Add vertices from a file to the vertex array.
   \subsection{Input parameters}
   \begin{itemize}
   \item[N] Number of points.
   \item[*nk] if nk$\rightarrow$n = N it is a ragged array of vertices.
          otherwise, it is an uninitialized array.
   \end{itemize}
   \subsection{Output parameters}
   \begin{itemize}
   \item[*nk] the list of vertices, including the ones read from the file.
   \end{itemize}
   \subsection{file}
   edge.c
**/

void ReadEdgeFile( int N, UIRaggedArray *nk ) {
   int i, j, v0, v1, fusion;
   unsigned int *pp, *cf;
   char **edgeset;
   FILE* in;

   fusion = ( nk->n == N );
   if( fusion ) {
      edgeset = InitCMatrix(N,N);
      ResetCMatrix(edgeset,N,N);
      for( i=0; i<N; i++ ) {
	 for( j=0; j<nk->c[i]; j++ )
	    edgeset[i][nk->p[i][j]] = 1;
      }
   }
   else {
      nk->n = N;
      nk->p = (unsigned int **) calloc(N, sizeof(unsigned int*));
      nk->c = (unsigned int *) calloc(N, sizeof(unsigned int));
      assure( nk->p && nk->c, "allocation failure " );
   }
   cf = InitUIVector(N);
   memcpy( cf, nk->c, N*sizeof(unsigned int) );

   in = fopen( GetParam( "EdgeFile" ), "r");
   assure( in, "edge file" );

   while( fscanf(in,"%d %d", &v0, &v1 ) == 2 ) {
      v0--;
      v1--;
      if( !(fusion && edgeset[v0][v1]) ) {
	 nk->c[ v0 ]++;
	 nk->c[ v1 ]++;
      }
   }
   fclose(in);
   
   if( fusion ) {
      for( i=0; i<N; i++ )
	if( nk->c[i] > cf[i] ) {
	   /* would have used realloc, but it has a bug on Linux */
	   pp = malloc( (nk->c[i])*sizeof(unsigned int) );
	   assure( pp, "malloc" );
	   memcpy( pp, nk->p[i], cf[i]*sizeof(unsigned int) );
	   free( nk->p[i] );
	   nk->p[i] = pp;
	}
   }
   else {
      for(i=0; i < N; i++) {
	nk->p[i]= calloc(nk->c[i], sizeof(unsigned int));
	assure( nk->p[i], "allocation failure" );
      }
   }
   
   in = fopen( GetParam( "EdgeFile" ), "r" );
   while( fscanf(in,"%d %d", &v0, &v1 ) == 2 ) {
      v0 --;
      v1 --;
      if( !(fusion && edgeset[v0][v1]) ) {
	 nk->p[ v0 ][ cf[v0] ] = v1;
	 nk->p[ v1 ][ cf[v1] ] = v0;
	 cf[ v0 ]++;
	 cf[ v1 ]++;
      }
   }
   fclose(in);
   
   if( fusion ) FreeCMatrix(edgeset,N);
   free(cf);
}

/* An auxiliary function for OrderEdges()                     */
/* returns the opposite result to that of uicompare in aux2.c */
int uicomp(const void *i, const void *j)
{ return (int)( *((unsigned int*)i) - *((unsigned int*)j) ); }

/**
   \section{OrderEdges}
   \subsection{Description}
   order the sub arrays of a ragged array in ascending order.
    \subsection{Input parameters}
   \begin{itemize}
    \item[NK] the ragged array to be ordered.
   \end{itemize}
    \subsection{Output parameters}
   \begin{itemize}
    \item[NK] The ordereded array.
         NK.p[i][j] $<$ NK.p[i][l] $\iff$ j $<$ l.
   \end{itemize}
   \subsection{Auxiliary function}
   int uicomp(const void *i, const void *j)
   \subsection{file}
   edge.c
**/    
void OrderEdges( UIRaggedArray *nk ) {
   int i;
   for( i=0; i<nk->n; i++ )
      qsort(nk->p[i],nk->c[i],sizeof(unsigned int),uicomp);   
}
	 
/**
    \section{InvertEdges}
    \subsection{Description}
    creates an inverted ragged array.
    \subsection{Input parameters}
   \begin{itemize}
    \item[NK] the ragged array to be inverted.
   \end{itemize}
    \subsection{Return value}
   \begin{itemize}
    \item[M] The inverted array.
        If k=NK.p[i][j] and i=NK.p[k][l] then M.p[i][j]=l.
   \end{itemize}
   \subsection{file}
   edge.c
**/    
UIRaggedArray InvertEdges(UIRaggedArray NK){
   UIRaggedArray M;
   int i,k,k0;

   M = InitUIRaggedArray( NK );

   for(i=0; i < NK.n; i++)
      for(k = 0; k < NK.c[i]; k++) {
	 k0 = 0;
	 while(NK.p[ NK.p[i][k] ][ k0 ] != i) k0++;
	 M.p[i][k] = k0;
      }

   return M;
}

/**
   \section{knn}
   \subsection{Description}
   Creates a mutual K nearest neighbours array.
   Fuses it with a minimal spanning tree if required.
   \subsection{Input parameters}
   \begin{itemize}
   \item[N] Number of points.
   \item[D] Dimension of vertex vector. \\
              $D=0 \longrightarrow$ use distances.
   \item[X] $D=0$: X is the distance matrix. \\
            $D>0$: X[i] is the $D$-dimentional vector ascociated with vertex i.
   \end{itemize}
   \subsection{Return value}
   \begin{itemize}
   \item[nk] nk.p[i] is the list of neighbours of vertex i.
   \end{itemize}
   \subsection{file}
   edge.c
**/

UIRaggedArray knn( int N, int D, double** X ) {
     
     double  *dist;		/* distances      */
     int    **MNV;	/* Nearest neighbours array */
     unsigned int    *indx;	/* auxiliar */
     UIRaggedArray   nk;        /* returned array */
     unsigned int **edg;        /* edges of mst */
     unsigned int *occ;
     
     unsigned int i,j,k,metric,K,similiarity,cand,gomstree;

     K = IGetParam( "KNearestNeighbours" );
     dist = calloc(N,sizeof(double));
     MNV = InitIMatrix(N,K);
     indx = InitUIVector(N);
     metric = ( GetParam( "InfMetric" ) == NULL );
     similiarity=(GetParam("DataIsInteraction")!=NULL);

     if(K>N)
       error("K > N ");
     
     
     /* Ordering of the neighbours - O(N^2 logN)*/
     for(i = 0; i < N; i++) {
       if(D != 0) {
	 for(j = 0; j < N; j++)
	   dist[j] = metric ?
	     Squared_Distance(D,X[i],X[j]) 
	     : Distance_Linf(D,X[i],X[j]);
       } else {
	 for(j = 0; j < N; j++)
	   dist[j] = X[i][j];
       }
       dist[i] = similiarity ? 0 : INF;

       DSortIndex(N,dist,indx);
       if (similiarity) 
	 for(j = 0; j < K; j++) 
	   MNV[i][j] = indx[N-j];
       else
	 for(j = 0; j < K; j++) 
	   MNV[i][j] = indx[j];
     }
     free(indx); free(dist);


     gomstree = (GetParam("MSTree")!=NULL);
     if( gomstree ) {
       edg = InitUIMatrix(N-1,2);
       mstree(N,D,X,edg);
     }

     /* Check for mutality - O(NK^2) */
     for (i=0;i<N;i++) {
       for(j=0;j<K;j++) {
	 if (MNV[i][j]<0)
	   continue;
	 cand = MNV[i][j];	/* candidate becomes ngbr if its mutual */
	 for(k=0;k<K && MNV[cand][k] != i;k++);
	 MNV[i][j]-=(MNV[i][j]+1)*(K==k); /* If the candidate is rejected */
	                                  /* its name is replaced by (-1). */
       }
     }


     /* Construction of the nk matrix O(NK)*/
     nk.n = N;
     nk.c = (unsigned int*)calloc(N,sizeof(unsigned int));
     nk.p = (unsigned int**)calloc(N,sizeof(unsigned int*));
     occ = (unsigned int*)calloc(N,sizeof(unsigned int));
     for(i = 0; i < N; i++) {     
       for (j=0;j<N;j++)
	 occ[j]=0; 
       for(j = 0; j < K; j++)
	 occ[MNV[i][j]]+=(MNV[i][j]>=0);
 
       if (gomstree) {
	 for(j=0;j<(N-1);j++)
	   if (edg[j][0]==i)
	     occ[edg[j][1]]++;
	   else if (edg[j][1]==i)
	     occ[edg[j][0]]++;
       }
	     
       for (j=0;j<N;j++)
	 nk.c[i]+=(occ[j]>0);
       nk.p[i] = (unsigned int*)calloc(nk.c[i],sizeof(unsigned int));
       for (k=0,j=0;j<N;j++)
	 if (occ[j]) nk.p[i][k++]=j;
     }
     if (gomstree)
       FreeUIMatrix(edg,N-1);
     FreeIMatrix(MNV,N);
     free(occ);
     
     return nk;
}


/* -------------------------------------------------------------------- */
/**
   \section{mstree}
   \subsection{Prim's Algorithm}
   \begin{enumerate}
   \item Set $i=0$, $S_0= \{u_0=s\}$, $L(u_0)=0$, and $L(v)=\inf$ for $v \neq u_0$. 
   If $|V| = 1$ then stop, otherwise go to step 2. 
   \item For each $v$ in $V \setminus S_i$, 
   replace $L(v)$ by $\min\{L(v), d_{v,u_i}\}$. 
   If $L(v)$ is replaced, put a label $(L(v), u_i)$ on $v$. 
   \item Find a vertex $v$ which minimizes $\{L(v) | v \in V \setminus S_i\}$, 
   say $u_{i+1}$. 
   \item Let $S_{i+1} = S_i \cup \{u_{i+1}\}$. 
   \item Replace $i$ by $i+1$. If $i=|V|-1$ then stop, otherwise go to step 2. 
   \end{enumerate}
   The time required by Prim's algorithm is $O(|V|^2)$. \\
   It can be reduced to $O(|E|\log|V|)$ if heap is used (but i didn't bother).
   \subsection{Input parameters}
   \begin{itemize}
   \item[N] number of points
   \item[d] distance matrix
   \end{itemize}
   \subsection{Output parameters}
   \begin{itemize}
   \item[**edg] the edges of the minimal spanning tree. Edge i is between
           vertices edg[i][0] and edg[i][1].
   \end{itemize}
   \subsection{file}
   edge.c
**/
/* -------------------------------------------------------------------- */
void mstree(int N,int D, double** X, unsigned int** edg) {
  int i,j,mi,u;
  float ml;
  int* V = (int*)calloc(N,sizeof(int));
  float* L = (float*)calloc(N,sizeof(float));
  int* label = (int*)calloc(N,sizeof(int));
  double d;
  char metric, similarity;

  metric = ( GetParam( "InfMetric" ) == NULL );
  similarity=(GetParam("DataIsInteraction")!=NULL);
  
  if (similarity) {
    fprintf(stderr,"\nWARNING: running mstree with similarity.\n");
  }

  for (i=0;i<(N-1);i++) {
    V[i] = i;
    L[i] = INF;
  }

  u = N-1;
  for (i=0;i<(N-1);i++) {
    ml = INF;
    for(j=0;j<(N-i-1);j++) {
      if (D)
        d = metric ?
          Squared_Distance(D,X[u],X[V[j]]) 
          : Distance_Linf(D,X[u],X[V[j]]);
      else
	      d = X[u][V[j]];
      if (d<=L[j]) {
        L[j] = d;
        label[j]=u;
      }
      if (L[j]<=ml) {
        ml = L[j];
        mi = j;
      }
    }
    edg[i][0] = label[mi]; 
    edg[i][1] = V[mi];
    u = V[mi];
    V[mi] = V[N-i-2];
    L[mi] = L[N-i-2];
    label[mi] = label[N-i-2];
  }
  free(V); free(L); free(label);
}
/* -------------------------------------------------------------------- */





