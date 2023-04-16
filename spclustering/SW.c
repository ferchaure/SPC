#include "SW.h"

int main(int argc, char *argv[])
{
   int N;                     /* Number of points                          */
   int Q;                     /* Number of Potts Spins; Si = 0,...,Q-1     */
   float T;                   /* Temperature                               */
   float Tmin, Tmax, dT;      /* Temp. min, max, and step                  */
   unsigned int *Spin;        /* Spin[i]= 0..Q-1 is the spin of point i    */
   UIRaggedArray NK;          /* Containns the neighboors of each point.   */
                              /* NK.n is the number of points and NK.c[i]  */
                              /* is the number of neighbours of point i.   */
                              /* NK[i][j], j=1..NK.c[i] are the labels of  */
                              /* neighboring points for each i=1..NK.n     */
   UIRaggedArray KN;          /* KN[i][k] = m, means that point i is the   */
                              /* m-th neighbor of point j = N[i][k]        */
   int D;                     /* Dimension of the point vectors, D=0 means */
                              /* that the distances are provided (instead  */
                              /* the coordinates of the vectors).          */
   double **X;                /* if D=0 X[i][j] is the distance between    */
                              /* points i and j. if else, it is the j-th   */
                              /* coordinate of the  i-th point.            */
   RaggedArray J;             /* J.p[i][j] is the interaction between the  */
                              /* i-th spin and its j-th neighbour          */
   RaggedArray P;             /* Deletion Probabilities for a satisfied    */
                              /* bond, i.e. if S[i] = S[j]. For an         */
                              /* unsatisfied bond deletion probability = 1 */
   CRaggedArray Bond;         /* Bond[i][k] takes value 1 (0) if bond      */
                              /* between spins i and its k-th neighbor is  */
                              /* frozen (deleted).                         */
   unsigned int *ClusterSize; /* ClusterSize[n] contains the number of    */
                              /* points belonging to cluster n in the      */
                              /* current configuration. The clusters are   */
                              /* ordered fron the biggest (n=1) to the     */
                              /* smallest                                  */
   unsigned int *Block;       /* Block[i] Is the number of the cluster to  */
                              /* which Spin[i] belongs                     */
   unsigned int *UIWorkSpc;   /* auxiliary work space                      */
   unsigned int *dgOldBlock;  /* dgOldBlock[i] Is the number of the cluster*/
                              /* to which Spin[i] belonged in the previous */
                              /* temperature after directed growth         */
   unsigned int *thOldBlock;  /* OldBlock[i] Is the number of the cluster  */
                              /* to which Spin[i] belonged in the previous */
                              /* temperature after thresholding            */
   UIRaggedArray CorrN;       /* Two Points Correlations comulant          */
   RARaggedArray FPCorr;      /* Four Points Correlations comulant         */
   float *Size1;              /* Sum of all clustersizes for a given T     */
   float *Size2;              /* Sum of all clustersizes^2 for a given T   */
   int nc;                    /* present number of clusters                */
   int nc1;                   /* cluster number comulant                   */
   int nb;                    /* present number of frozen bonds            */
   int ncy, cyc;              /* cyles counter, total number of cycles      */
   int nT;                    /* Temp. step counter                         */
   float th_MIN, th_MAX, dth; /* delta-threshold min, max and step          */
   float thN;                 /* the current threshold (loop parameter)     */
   int nth;                   /* numbers for the different threshold files  */
   float SWfract;             /* the fraction SW sweeps for which averages  */
                              /* are calculated. The first (1-SWfract)*cyc  */
                              /* sweeps are discarded.                      */
   float *mag;                /* Actual Magnetization                       */
   float *M1, *M2;            /* Magnetization cumulants                    */
   float *xi;                 /* Sussceptibility at the current temp.       */
   double e, E, EE;           /* Energy cumulants                           */
   int save_suscept;          /* If set, save and report susceptibily and   */
                              /* magnetization data                         */
   int save_averages;         /* If set, save and report SW averages        */
   int n_cols;                /* N times connecting points from different   */
                              /* labels in previous temp. at theta stage    */
   int i;                     /* auxiliary loop index                       */
   char cflag = 1;            /* Continue run flag */

   printf("\nSPC v2.1 (14/2/2000) -- efficient KNN and mstree");
   printf("\nSPC v2.0.1 (1/12/2000) -- StopRunAtBreak option ");
   printf("\nSPC v2.0 (1/11/1999) -- new param file + revised data structures ");
   printf("\n-------------------------------------------------------------------\n");
   if (argc < 2)
   {
      printf("Usage: %s [Parameter file]\n", argv[0]);
      exit(1);
   }
   DefaultParam();
   ReadParam(argc, argv);
   /* spelling mistakes I made. better be removed in the future... */
   if (!GetParam("Dimensions"))
      SetParam("Dimensions", GetParam("Dimentions"));
   if (GetParam("WriteLables"))
      SetParam("WriteLabels", NULL);
   /* .............................................................*/
   CheckParam();

   if (GetParam("Timing"))
      start_timer();

   if (GetParam("ForceRandomSeed"))
      i = IGetParam("ForceRandomSeed");
   else
      i = time(NULL) % RAND_MAX % INT_MAX;
   srand(i);
   ISetParam("RandomSeed", i);

   N = IGetParam("NumberOfPoints");
   Q = IGetParam("PottsSpins");
   SWfract = FGetParam("SWFraction");
   cyc = IGetParam("SWCycles");
   Tmin = FGetParam("MinTemp");
   Tmax = FGetParam("MaxTemp");
   dT = FGetParam("TempStep");
   D = IGetParam("Dimensions");
   save_averages = (GetParam("SaveAverages") != NULL);
   save_suscept = (GetParam("SaveSuscept") != NULL);
   if ((dth = FGetParam("ThresholdStep")) != 0.0)
   {
      th_MIN = FGetParam("ThresholdMin");
      th_MAX = FGetParam("ThresholdMax");
   }
   else
      dth = th_MIN = th_MAX = FGetParam("ThresholdTheta");

   if (D == 0)
      X = InitDMatrix(N, N);
   else
      X = InitDMatrix(N, D);
   ReadData(N, D, X);

   NK.n = 0;
   if (GetParam("KNearestNeighbours") || GetParam("MSTree"))
      NK = knn(N, D, X);
   if (GetParam("EdgeFile"))
      ReadEdgeFile(N, &NK);
   OrderEdges(&NK); /* Edges *must* be ordered when calling SetBond() */
   KN = InvertEdges(NK);
   if (GetParam("WriteEdges"))
      WriteEdges(NK);
   J = EdgeDistance(D, NK, X);
   assure(IGetParam("NumberOfEdges") > 0, "no edges");

   FreeDMatrix(X, N);

   if (!GetParam("DataIsInteraction"))
      DistanceToInteraction(J, NK, KN);
   FSetParam("AverageInteraction", AverageInteraction(J));
   if (GetParam("PrintParam"))
      PrintParam();
   /* Memory allocations: */
   CorrN = InitUIRaggedArray(NK);
   Bond = InitCRaggedArray(NK);
   P = InitRaggedArray(NK);
   ClusterSize = InitUIVector(N);
   Block = InitUIVector(N);
   if (save_averages)
   {
      Size1 = InitVector(N);
      Size2 = InitVector(N);
   }
   if (save_suscept)
   {
      mag = InitVector(Q);
      M1 = InitVector(Q);
      M2 = InitVector(Q);
      xi = InitVector(Q);
   }
   if (GetParam("FourPointCorr") || GetParam("WriteFPSum"))
      FPCorr = InitRARaggedArray(NK);
   else
      FPCorr.n = 0;
   UIWorkSpc = InitUIVector((2 * N > Q) ? 2 * N : Q); /* bounds on UIWorkSpc size:
                                                         >=Q for magnetization
                                >=2N for OrderingClusters  */
   Spin = InitUIVector(N);
   InitialSpinConfig(N, Spin, Q);

   dgOldBlock = InitUIVector(N);
   thOldBlock = InitUIVector(N);
   memset(dgOldBlock, 0, N * sizeof(unsigned int));
   memset(thOldBlock, 0, N * sizeof(unsigned int));
   if (GetParam("PrevTempFile"))
      ReadPrevTempFiles(thOldBlock, dgOldBlock, N);

   if (GetParam("Timing"))
      PrintTime(-1.0);

   /*********************** T LOOP **********************/
   for (T = Tmin, nT = 0; T <= Tmax && cflag; nT++, T = GetParam("TempStepMul") ? T * dT : T + dT)
   {

      /* Memory reset: */
      ResetUIRaggedArray(CorrN);
      ResetCRaggedArray(Bond);
      ResetRaggedArray(P);
      if (FPCorr.n)
         ResetRARaggedArray(FPCorr);
      if (save_averages)
      {
         nc1 = 0;
         E = EE = 0;
         memset(Size1, 0, N * sizeof(float));
         memset(Size2, 0, N * sizeof(float));
      }
      if (save_suscept)
      {
         memset(M1, 0, Q * sizeof(float));
         memset(M2, 0, Q * sizeof(float));
      }

      DeletionProbabilities(T, J, P);

      /* Transient of Monte Carlo (not included in averages) */
      for (i = 0; i < cyc * (1. - SWfract); i++)
      {
         nb = SetBond(P, Spin, Bond, NK, KN);
         nc = Coarsening(Bond, Block, NK, ClusterSize, UIWorkSpc);
         NewSpinConfig(N, Spin, Block, nc, Q, UIWorkSpc);
      }

      /***************** START MC LOOP ********************/
      for (ncy = 0; ncy <= cyc * SWfract; ncy++)
      {

         nb = SetBond(P, Spin, Bond, NK, KN);
         nc = Coarsening(Bond, Block, NK, ClusterSize, UIWorkSpc);
         NewSpinConfig(N, Spin, Block, nc, Q, UIWorkSpc);

         GlobalCorrelation(CorrN, NK, Block);
         if (FPCorr.n || GetParam("WriteFPSum"))
            FourPointCorrelation(FPCorr, NK, Block);

         if (save_suscept || save_averages)
            OrderClusterSize(nc, ClusterSize);
         if (save_averages)
         {
            nc1 += nc;
            e = Energy(Block, J, NK);
            E += e;
            EE += e * e;
            for (i = 0; i < nc; i++)
            {
               Size1[i] += (float)(ClusterSize[i]);
               Size2[i] += (float)(ClusterSize[i] * ClusterSize[i]);
            }
         }
         if (save_suscept)
         {
            Magnetization(N, Q, nc, ClusterSize, mag, UIWorkSpc);
            for (i = 0; i < Q; i++)
            {
               M1[i] += mag[i];
               M2[i] += mag[i] * mag[i];
            }
         }
      } /********************* END MC LOOP *********************/
      if (save_averages)
      {
         ClusterAverage(ncy, N, Size1, Size2);
         PrintAverages(nT, T, E / ncy, EE / ncy, (float)nc1 / (float)ncy, Size1);
      }

      if (save_suscept)
      {
         Susceptibility(Q, ncy, M1, M2, xi);
         PrintMagnet(nT, T, M1, xi);
      }

      if (GetParam("WriteCorFile"))
         NNPrintCorrN(nT, T, CorrN, ncy, NK);
      if (FPCorr.n)
         PrintFPointCorr(nT, T, FPCorr, NK, ncy);
      if (GetParam("WriteFPSum"))
         PrintFPSum(nT, J, CorrN, FPCorr, NK, ncy);

      if (GetParam("Threshold"))
      {
         nth = 0;
         for (thN = th_MIN; thN <= th_MAX; thN += dth)
         {
            nth++;
            n_cols = 0;
            nc = Thresholding(ncy, thN, CorrN, NK, Bond, Block, ClusterSize,
                              thOldBlock, &n_cols, UIWorkSpc);
            PrintSizes(".th_", nth, nT, T, nc, ClusterSize, n_cols);
            if (GetParam("WriteLabels"))
               WriteLabels(".th_", nth, nT, T, N, Block);
         }
         memcpy(thOldBlock, Block, N * sizeof(unsigned int));
      }

      /* threshod + directed growth */
      if (GetParam("DirectedGrowth") || GetParam("StopRunAtBreak"))
      {
         nth = 0;
         for (thN = th_MIN; thN <= th_MAX; thN += dth)
         {
            nth++;
            nc = DirectedGrowth(ncy, thN, CorrN, NK, KN, Bond, Block, ClusterSize,
                                dgOldBlock, thOldBlock, UIWorkSpc);
            /* Notice that above thOldBlock is the new thBlocks but */
            /* it is OK to use them */
            if (IGetParam("StopRunAtBreak"))
            {
               int srcnt;
               for (srcnt = 0; ClusterSize[srcnt] >= IGetParam("StopRunClusterSize") && srcnt < nc; srcnt++)
                  ;
               cflag = (srcnt < IGetParam("StopRunAtBreak"));
            }
            PrintSizes(".dg_", nth, nT, T, nc, ClusterSize, 0);
            if (GetParam("WriteLabels"))
               WriteLabels(".dg_", nth, nT, T, N, Block);
         }
         if (!GetParam("Threshold"))
            memcpy(thOldBlock, Block, N * sizeof(unsigned int));
         memcpy(dgOldBlock, Block, N * sizeof(unsigned int));
      }

      if (GetParam("Timing"))
         PrintTime(T);

   } /******************** END OF T LOOP ***********************/

   FreeUIRaggedArray(CorrN);
   FreeCRaggedArray(Bond);
   FreeRaggedArray(P);
   free(ClusterSize);
   free(Block);
   free(UIWorkSpc);
   if (save_averages)
   {
      free(Size1);
      free(Size2);
   }
   if (save_suscept)
   {
      free(mag);
      free(M1);
      free(M2);
      free(xi);
   }
   if (FPCorr.n)
      FreeRARaggedArray(FPCorr);

   FreeUIRaggedArray(NK);
   FreeUIRaggedArray(KN);
   FreeRaggedArray(J);
   free(Spin);
   free(dgOldBlock);
   free(thOldBlock);

   return (0);
}
