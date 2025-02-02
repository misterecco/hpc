#include <omp.h>
#include <iostream>
#include <iomanip>

// #define STEPS 1000
// #define THREADS 16 //you can also use the OMP_NUM_THREADS environmental variable

double power(double x, long n) {
    if (n == 0) {
        return 1;
    }

    return x * power(x, n - 1);
}

double calcPi(long n) {
    if (n < 0) {
        return 0;
    }

    return 1.0 / power(16, n)
           * (4.0 / (8 * n + 1.0)
              - 2.0 / (8 * n + 4.0)
              - 1.0 / (8 * n + 5.0)
              - 1.0 / (8 * n + 6.0))
           + calcPi(n - 1);
}

double powerParallelReduction(double x, long n) {
    double power = 1;
      
    #pragma omp parallel for reduction(*:power)
    for (long i = 1; i < n+1; i++) {
        power *= x;
    }

    return power;
}

double powerParallelCritical(double x, long n) {
    double power = 1;
      
    #pragma omp parallel for
    for (long i = 1; i < n+1; i++) {
        #pragma omp critical
        power *= x;
    }

    return power;
}

double calcPiParallelReduction(long n) {
    double pi = 0;
      
    #pragma omp parallel for reduction(+:pi)
    for (long i = 0; i < n; i++) {
        pi += 1.0 / powerParallelReduction(16, i)
           * (4.0 / (8 * i + 1.0) - 2.0 / (8 * i + 4.0)
            - 1.0 / (8 * i + 5.0) - 1.0 / (8 * i + 6.0));
    }

    return pi;
}

double calcPiParallelCritical(long n) {
    double pi = 0;
      
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
        double partial = 1.0 / powerParallelCritical(16, i)
           * (4.0 / (8 * i + 1.0) - 2.0 / (8 * i + 4.0)
            - 1.0 / (8 * i + 5.0) - 1.0 / (8 * i + 6.0));
        #pragma omp critical
        pi += partial;
    }

    return pi;
    // PUT IMPLEMENTATION HERE
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " steps" << std::endl;
    return 1;
  }      

  long steps = atol(argv[1]);

  double startTime, sequentialTime, reductionTime, criticalTime;

  startTime = omp_get_wtime();
    std::cout << std::setprecision(10) << calcPi(steps) << std::endl;
  sequentialTime = omp_get_wtime() - startTime;

  startTime = omp_get_wtime();
    std::cout << std::setprecision(10) << calcPiParallelReduction(steps) << std::endl;
  reductionTime = omp_get_wtime() - startTime;

  startTime = omp_get_wtime();
    std::cout << std::setprecision(10) << calcPiParallelCritical(steps) << std::endl;
  criticalTime = omp_get_wtime() - startTime;

    std::cout << "TIMES: sequential: " << sequentialTime << ", reduction: " 
      << reductionTime << ", critical: " << criticalTime << std::endl;

    std::cout << "SPEEDUPs: reduction: " << sequentialTime / reductionTime
      << ", critical: " << sequentialTime / criticalTime << std::endl;

    return 0;
}
