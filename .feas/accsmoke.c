#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <openacc.h>

#pragma acc routine seq
static double f(double x) { return sqrt(x) * 2.0 + 1.0; }

int main(void) {
  int n = 100000;
  printf("acc devices (nvidia): %d\n", acc_get_num_devices(acc_device_nvidia));
  double *x = malloc(n * sizeof(double)), *y = malloc(n * sizeof(double));
  for (int i = 0; i < n; i++) { x[i] = i; y[i] = 1.0; }
  double sum = 0.0, hist[16] = {0};
  #pragma acc parallel loop copyin(x[0:n]) copy(y[0:n]) reduction(+:sum) copy(hist[0:16])
  for (int i = 0; i < n; i++) {
    y[i] = 0.5 * f(x[i]) + y[i];
    sum += x[i];
    #pragma acc atomic
    hist[i % 16] += 1.0;
  }
  int bad = 0;
  for (int i = 0; i < n; i++) if (fabs(y[i] - (0.5 * (sqrt((double)i) * 2.0 + 1.0) + 1.0)) > 1e-9) bad++;
  double want = (double)n * (n - 1) / 2.0;
  for (int k = 0; k < 16; k++) if (hist[k] != (double)(n / 16)) bad++;
  printf("sum %.1f want %.1f, bad %d, y[100]=%f\n", sum, want, bad, y[100]);
  if (bad || sum != want) { printf("FAIL\n"); return 1; }
  printf("PASS\n");
  return 0;
}
