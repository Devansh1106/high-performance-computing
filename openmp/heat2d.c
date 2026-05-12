#include <stdio.h>
#include <omp.h>

#define NX 60
#define NY 60
#define STEPS 100   // total time steps (you can adjust)

int main() {

    double u[NX][NY], u_new[NX][NY];

    double dx = 1.0/(NX-1);
    double dt = 0.125;
    double alpha = 2.0;

    double coeff = alpha * dt / (dx*dx);

    // initial condition
    for (int i=0;i<NX;i++)
        for (int j=0;j<NY;j++)
            u[i][j] = 0;

    // boundary conditions
    for (int i=0;i<NX;i++) {
        u[i][0] = 0;
        u[i][NY-1] = 75;
    }

    for (int j=0;j<NY;j++) {
        u[0][j] = 0;
        u[NX-1][j] = 0;
    }

    double start = omp_get_wtime();

    for (int t=0;t<STEPS;t++) {

        // Jacobi update
        #pragma omp parallel for collapse(2)
        for (int i=1;i<NX-1;i++) {
            for (int j=1;j<NY-1;j++) {
                u_new[i][j] = u[i][j] + coeff * (
                    u[i+1][j] + u[i-1][j] +
                    u[i][j+1] + u[i][j-1] -
                    4*u[i][j]
                );
            }
        }

        // copy back
        #pragma omp parallel for collapse(2)
        for (int i=1;i<NX-1;i++)
            for (int j=1;j<NY-1;j++)
                u[i][j] = u_new[i][j];
    }

    double end = omp_get_wtime();

    printf("CPU time = %f\n", end - start);

    // write to file
    FILE *fp = fopen("output.txt", "w");

    for (int i=0;i<NX;i++) {
        for (int j=0;j<NY;j++)
            fprintf(fp, "%f ", u[i][j]);
        fprintf(fp, "\n");
    }

    fclose(fp);

    return 0;
}