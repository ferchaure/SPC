#include "SW.h"

void ReadData(int N, int D, double **X)
{
   char *file;
   int i, d, j, dij;
   double t;
   FILE *in;

   dij = (GetParam("DataIsInteraction") != NULL);
   file = GetParam("DataFile");
   in = fopen(file, "r");
   assure(in, "file not found");

   if (D > 0)
      for (i = 0; i < N; i++)
         for (d = 0; d < D; d++)
            fscanf(in, "%lf", &X[i][d]);
   else
   {
      if (GetParam("DataIsMatrix"))
      {
         for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
               assure(fscanf(in, "%lf", &X[i][j]) == 1, "not enough elements");
      }
      else
      {
         for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
               X[i][j] = dij ? 0 : INF;
         while (fscanf(in, "%d %d %lf", &i, &j, &t) == 3)
         {
            i--, j--;
            assure(i < N && j < N, "data error");
            X[j][i] = X[i][j] = t;
         }
      }
   }
   fclose(in);
}

/* ----------------------------------------------------------------------- */
/* Read label files of prev temperature basename.th_01.lab, .dg_01.lab     */
/* and update thOldBlock, dgOldBlock.
/* ----------------------------------------------------------------------- */
void ReadPrevTempFiles(unsigned int *thOldBlock,
                       unsigned int *dgOldBlock, int N)
{
   char fname[PARAM_MAX_FIELD + 10], *basename;
   FILE *fp = NULL;
   int i;
   int dummy;
   float mintemp, t;

   assure(basename = GetParam("PrevTempFile"), "parameter not set");
   mintemp = FGetParam("MinTemp");
   strcpy(fname, basename);
   strcat(fname, ".th_01.lab");
   assure(fp = fopen(fname, "r"), "Could not find prev. temp. label files");
   while (fscanf(fp, "%d", &dummy) == 1)
   {
      fscanf(fp, "%f", &t);
      if (t >= mintemp)
         break;
      for (i = 0; i < N; i++)
      {
         fscanf(fp, "%d", &dummy);
         thOldBlock[i] = (unsigned int)dummy;
      }
   }
   fclose(fp);

   strcpy(fname, basename);
   strcat(fname, ".dg_01.lab");
   assure(fp = fopen(fname, "r"), "Could not find prev. temp. label files");
   while (fscanf(fp, "%d", &dummy) == 1)
   {
      fscanf(fp, "%f", &t);
      if (t >= mintemp)
         break;
      for (i = 0; i < N; i++)
      {
         fscanf(fp, "%d", &dummy);
         dgOldBlock[i] = (unsigned int)dummy;
      }
   }
   fclose(fp);

   return;
}

/* -------------------------------------------------------------------- */
/* The averaged sizes of the SW-clusters are printed out to a file      */
/* -------------------------------------------------------------------- */
void PrintAverages(int nT, float T, double e, double ee, float nc, float *Size)
{
   int i, NS, MinSize;
   FILE *out;
   char file[PARAM_MAX_FIELD + 10], ext[10];

   NS = IGetParam("ClustersReported");
   MinSize = IGetParam("ClusterMinSizeReported");

   strcpy(file, GetParam("OutFile"));
   strcpy(ext, ".ave");
   strcat(file, ext);
   out = fopen(file, "a");

   fprintf(out, "%3d ", nT);      /* serial number */
   fprintf(out, "%8.5f    ", T);  /* temperature */
   fprintf(out, "%8.5f    ", e);  /* energy */
   fprintf(out, "%8.5f    ", ee); /* energy square */
   fprintf(out, "%6.3f    ", nc); /* average number of blocks */
   for (i = 0; i < NS || (MinSize && (Size[i] >= MinSize)); i++)
      fprintf(out, "%5.0f   ", Size[i]);
   fprintf(out, "\n");
   fclose(out);

   printf("ave: ");
   printf("%8.5f    ", T);
   printf("%8.5f    ", e);
   printf("%8.5f    ", ee);
   printf("%8.3f    ", nc);
   for (i = 0; i < NS || (MinSize && (Size[i] >= MinSize)); i++)
      printf("%5.0f   ", Size[i]);
   printf("\n");
}

void PrintMagnet(int nT, float T, float *M, float *xi)
{
   int i, suscol, Q;
   FILE *out;
   char file[PARAM_MAX_FIELD + 10];

   suscol = IGetParam("SusceptColors");
   Q = IGetParam("PottsSpins");
   if (Q < suscol)
      suscol = Q;

   strcpy(file, GetParam("OutFile"));
   strcat(file, ".mag");
   out = fopen(file, "a");

   fprintf(out, "%3d ", nT);
   fprintf(out, "%8.5f    ", T);
   for (i = 0; i < suscol; i++)
      fprintf(out, "%8.6f   ", M[i]);
   for (i = 0; i < suscol; i++)
      fprintf(out, "%8.6f   ", xi[i]);
   fprintf(out, "\n");
   fclose(out);
}

void NNPrintCorrN(int nT, float T, UIRaggedArray CorrN, int ncy,
                  UIRaggedArray NK)
{
   int i, k;
   FILE *out;
   char file[PARAM_MAX_FIELD + 10];

   strcpy(file, GetParam("OutFile"));
   strcat(file, ".cor");
   out = fopen(file, "a");

   fprintf(out, "%3d", nT);
   fprintf(out, "\t%8.5f", T);
   for (i = 0; i < CorrN.n; i++)
      for (k = 0; k < CorrN.c[i]; k++)
         if (NK.p[i][k] > i)
            fprintf(out, "\t%8.5f", (float)CorrN.p[i][k] / (float)ncy);

   fprintf(out, "\n");
   fclose(out);
   return;
}

/*******************************/
/* prints 4-point Correlations */
/*******************************/
void PrintFPointCorr(int nT, float T, RARaggedArray FPCorr,
                     UIRaggedArray NK, int ncy)
{
   int i, k, i1, k1;
   FILE *out;
   char file[PARAM_MAX_FIELD + 10];

   strcpy(file, GetParam("OutFile"));
   strcat(file, ".4pc");
   out = fopen(file, "a");
   for (i = 0; i < FPCorr.n; i++)
      for (k = 0; k < FPCorr.c[i]; k++)
         if (NK.p[i][k] > i)
            for (i1 = i + 1; i1 < NK.n; i1++)
               for (k1 = 0; k1 < FPCorr.c[i1]; k1++)
                  if (NK.p[i1][k1] > i1)
                     fprintf(out, "%8.5f     ",
                             (float)((FPCorr.p[i][k]).p[i1][k1]) / (float)ncy);
   fprintf(out, "\n");
   fclose(out);
   return;
}

float chkinteg(float q)
{
   if (q > 0)
      return (q);
   if (q < (-0.05))
      printf("Warning: Negative Co-Correlation: %f.\n", q);
   return (0.0);
}

void PrintFPSum(int nT, RaggedArray J, UIRaggedArray Corr, RARaggedArray FPCorr,
                UIRaggedArray NK, int ncy)
{
   int i, k;
   int i1, k1;
   float s = 0, dum;
   FILE *out;
   char file[80];

   strcpy(file, GetParam("OutFile"));
   strcat(file, ".4psum");
   out = fopen(file, "a");
   for (i = 0; i < FPCorr.n; i++)
      for (k = 0; k < FPCorr.c[i]; k++)
      {
         if (NK.p[i][k] > i)
         {
            s = 0;
            for (i1 = i + 1; i1 < NK.n; i1++)
               for (k1 = 0; k1 < FPCorr.c[i1]; k1++)
               {
                  if (NK.p[i1][k1] > i1)
                  {
                     dum = (((float)(FPCorr.p[i][k]).p[i1][k1]) -
                            ((float)Corr.p[i][k]) * ((float)Corr.p[i1][k1]) / ((float)ncy)) /
                           ((float)ncy);
                     dum = chkinteg(dum);
                     s += J.p[i1][k1] * dum;
                  }
               }
            for (i1 = 0; i1 < i; i1++)
               for (k1 = 0; k1 < FPCorr.c[i1]; k1++)
               {
                  if (NK.p[i1][k1] > i1)
                  {
                     dum = (((float)(FPCorr.p[i][k]).p[i1][k1]) -
                            ((float)Corr.p[i][k]) * ((float)Corr.p[i1][k1]) / ((float)ncy)) /
                           ((float)ncy);
                     dum = chkinteg(dum);
                     s += J.p[i1][k1] * dum;
                  }
               }
            fprintf(out, "%8.5f     ", s);
         }
      }
   fprintf(out, "\n");
   fclose(out);
   return;
}

/* -------------------------------------------------------------------- */
/* The sizes of the SW-clusters are printed out to a file               */
/* -------------------------------------------------------------------- */
void PrintSizes(char *pe, int nth, int nT, float T, int nc,
                unsigned int *Size, int n_cols)
{
   int i, NS, MinSize;
   FILE *out;
   char file[PARAM_MAX_FIELD + 10], thl[3], ext[8];

   strcpy(file, GetParam("OutFile"));
   strcpy(ext, pe);
   assure(nth <= 99, "nth too large");
   thl[0] = '0' + (nth / 10);
   thl[1] = '0' + (nth % 10);
   thl[2] = 0;
   strcat(ext, thl);
   strcat(file, ext);

   NS = IGetParam("ClustersReported");
   MinSize = IGetParam("ClusterMinSizeReported");

   out = fopen(file, "a");

   fprintf(out, "%3d ", nT);
   fprintf(out, "%8.5f  ", T);
   fprintf(out, "%8d  ", n_cols);
   fprintf(out, "%5d    ", nc);
   for (i = 0; i < NS || (MinSize && (Size[i] >= MinSize)); i++)
      fprintf(out, "%5d   ", Size[i]);
   fprintf(out, "\n");

   fclose(out);
}

void WriteLabels(char *pe, int nth, int nT, float T, int N, unsigned int *Block)
{
   int i;
   FILE *out;
   char file[PARAM_MAX_FIELD + 20], thl[3], ext[15];

   strcpy(file, GetParam("OutFile"));
   strcpy(ext, pe);
   assure(nth <= 99, "nth too large");
   thl[0] = '0' + (nth / 10);
   thl[1] = '0' + (nth % 10);
   thl[2] = 0;
   strcat(ext, thl);
   strcat(ext, ".lab");
   strcat(file, ext);

   out = fopen(file, "a");

   fprintf(out, "%3d ", nT);
   fprintf(out, "%8.5f  ", T);
   for (i = 0; i < N; i++)
      fprintf(out, "%4d ", Block[i]);
   fprintf(out, "\n");

   fclose(out);
}

void ReadParamFromStr(char *param_str)
{
   char s[PARAM_MAX_FIELD + 50],
       n[PARAM_MAX_FIELD + 1], ac;
   int l;
   s[0] = 0;
   ac = 0;
   while (sscanf(param_str, "%s", s) != EOF)
   {
      l = strlen(s);
      param_str = param_str + (l + 1);
      if (!(l))
         continue;
      if (ac == ':')
      {
         if (s[l - 1] != ':' && s[l - 1] != '~' && s[l - 1] != '|')
            SetParam(n, s);
         else
            SetParam(n, NULL);
      }
      ac = s[l - 1];
      s[l - 1] = 0;
      if (ac == ':')
         strcpy(n, s);
      if (ac == '|')
         SetParam(s, NULL);
      if (ac == '~')
         UnsetParam(s);
      s[0] = 0;
   }
   if (ac == ':')
      SetParam(n, NULL);
}

void ReadParam(int argc, char *argv[])
{
   char file[PARAM_MAX_FIELD + 100], s[PARAM_MAX_FIELD + 50],
       n[PARAM_MAX_FIELD + 1], ac;
   int l;
   FILE *in;
   if (argc < 2)
      strcpy(file, "SW.ini");
   else
      strcpy(file, argv[1]);
   in = fopen(file, "r");
   assure(in, "parameter file error");
   s[0] = 0;
   ac = 0;
   while (fscanf(in, "%s", s) != EOF)
   {
      if (!(l = strlen(s)))
         continue;
      if (ac == ':')
      {
         if (s[l - 1] != ':' && s[l - 1] != '~' && s[l - 1] != '|')
            SetParam(n, s);
         else
            SetParam(n, NULL);
      }
      ac = s[l - 1];
      s[l - 1] = 0;
      if (ac == ':')
         strcpy(n, s);
      if (ac == '|')
         SetParam(s, NULL);
      if (ac == '~')
         UnsetParam(s);
      s[0] = 0;
   }
   if (ac == ':')
      SetParam(n, NULL);
}

void PrintParam()
{
   FILE *out;
   char file[PARAM_MAX_FIELD + 10];
   strcpy(file, GetParam("OutFile"));
   strcat(file, ".param");
   out = fopen(file, "w");
   assure(out, "out param file");
   fprint_param(out);
   fclose(out);
   fprint_param(stdout);
}

void WriteEdges(UIRaggedArray nk)
{
   int i, k;
   char file[PARAM_MAX_FIELD + 30], ck[15];
   FILE *out;

   sprintf(ck, "%d.edges", IGetParam("KNearestNeighbours"));
   if (GetParam("EdgeFile"))
      strcpy(file, GetParam("EdgeFile"));
   else
      strcpy(file, GetParam("OutFile"));
   if (GetParam("MSTree"))
      strcat(file, ".mst");
   else
      strcat(file, ".K");
   strcat(file, ck);

   out = fopen(file, "w");
   assure(out, "edge file");

   for (i = 0; i < nk.n; i++)
      for (k = 0; k < nk.c[i]; k++)
         if (nk.p[i][k] > i)
            fprintf(out, "%d\t%d\n", i + 1, nk.p[i][k] + 1);

   fclose(out);
}

void PrintTime(float T)
{
   double user_time, total_user_time, real_time, total_real_time;
   FILE *out;
   char file[PARAM_MAX_FIELD + 10], temp[20];

   strcpy(file, GetParam("OutFile"));
   strcat(file, ".timing");
   out = fopen(file, "a");
   assure(out, "timing file error");

   if (T < 0)
      strcpy(temp, "initialization");
   else
      sprintf(temp, "T=%f", T);
   get_timer(&user_time, &total_user_time, &real_time, &total_real_time);
   fprintf(out, "Time for %s:\tuser time %.2lf sec\treal time %d sec\n",
           temp, user_time, (int)real_time);

   fclose(out);
}
