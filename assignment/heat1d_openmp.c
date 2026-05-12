#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#define pi 3.14159265

int main ()
{
    int n = 2048;
    double* temp = NULL;
    double* u = (double*) calloc(n, sizeof(double)); // boundary condition 0.0 is already satisfied
    double* sol = (double*) calloc(n, sizeof(double));
    double* exact_sol = (double*) calloc(n, sizeof(double));
    double h = 1.0/(n-1);
    double c = 0.3;
    double alpha = 0.25;
    double dt = (alpha * h * h) / c;
    double final_t = 0.2;
    printf("Time step size is %0.12f\n", dt);
    printf("Step size in space is %0.12f\n", h);
    printf("Alpha is %f\n", alpha);
    
    int iter = round(final_t/dt);
    double global_sum = 0.0;

    double start_time = omp_get_wtime();
    #pragma omp parallel
    {
        double local_sum = 0.0;
        #pragma omp for 
        for(int i = 0; i < n; i++){
            double x = i*h;
            u[i] = 0.1*sin(pi * x) + sin(2 * pi * x);
            exact_sol[i] = 0.1*sin(pi * x) * exp(-0.3 * pi * pi * final_t) + sin(2 * pi * x) * exp(-1.2 * pi * pi * final_t);
        }

        for(int j = 0; j < iter; j++){
            #pragma omp for
            for(int i = 1; i < n-1; i++){
                sol[i] = alpha*u[i+1] + alpha*u[i-1] + (1-2*alpha)*u[i];
            }
            #pragma omp single
            {
                temp = u;
                u = sol;
                sol = temp;
            }
        }
        // L2 error
        #pragma omp for
        for (int i = 0; i < n; i++){
            double temp1 = exact_sol[i] - u[i];
            local_sum += temp1 * temp1;
        }

        #pragma omp critical
        global_sum += local_sum;
    }
    double end_time = omp_get_wtime();
    printf("Execution time: %0.12f seconds\n", end_time - start_time);

    printf("L^2 error is %0.12f\n", sqrt(global_sum*h));

    free(u); free(sol); free(exact_sol);
    return 0;
}