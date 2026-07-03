#include "timer.h"
#include <stdlib.h>
#include <time.h>

static clock_t begin_clock, save_clock;
static time_t begin_time, save_time;

void start_timer() {
  begin_clock = save_clock = clock();
  begin_time = save_time = time(NULL);
}

void get_timer(double *user_time, double *total_user_time, double *real_time,
               double *total_real_time) {
  clock_t clk = clock();
  time_t tm = time(NULL);

  *user_time = (clk - save_clock) / ((double)CLOCKS_PER_SEC);
  *total_user_time = (clk - begin_clock) / ((double)CLOCKS_PER_SEC);
  *real_time = difftime(tm, save_time);
  *total_real_time = difftime(tm, begin_time);
  save_clock = clk;
  save_time = tm;
}
