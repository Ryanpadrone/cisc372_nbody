#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include "vector.h"
#include "config.h"

// Host globals are defined in nbody.c and declared in vector.h
extern vector3 *hVel, *d_hVel;
extern vector3 *hPos, *d_hPos;
extern double *mass;

// Device-only arrays used in this file
static double  *d_mass   = NULL;
static vector3 *d_accels = NULL;  // flattened NUMENTITIES x NUMENTITIES matrix

static int gpu_initialized = 0;

// Simple CUDA error checking helper
static void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// Allocate device memory once
static void initGpu(int n) {
    if (gpu_initialized) return;

    // Positions and velocities on device use the extern pointers
    checkCuda(cudaMalloc((void**)&d_hPos,  n * sizeof(vector3)), "cudaMalloc d_hPos");
    checkCuda(cudaMalloc((void**)&d_hVel,  n * sizeof(vector3)), "cudaMalloc d_hVel");

    // Mass array on device
    checkCuda(cudaMalloc((void**)&d_mass,  n * sizeof(double)),  "cudaMalloc d_mass");

    // Acceleration matrix: n x n of vector3
    checkCuda(cudaMalloc((void**)&d_accels, n * n * sizeof(vector3)), "cudaMalloc d_accels");

    gpu_initialized = 1;
}

/*
 * Kernel 1: compute pairwise accelerations accels[i][j]
 *
 * This matches your original CPU logic:
 *   if (i == j) accel = (0,0,0)
 *   distance = hPos[i] - hPos[j]
 *   magnitude_sq = |distance|^2
 *   magnitude    = sqrt(magnitude_sq)
 *   accelmag = -GRAV_CONSTANT * mass[j] / magnitude_sq
 *   accel = accelmag * distance / magnitude
 *
 * Stored in a flattened array: accels[i * n + j]
 */
__global__ void computeAccelsKernel(vector3 *pos,
                                    double  *mass,
                                    vector3 *accels,
                                    int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // source index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // target index

    if (i >= n || j >= n) return;

    int idx = i * n + j;

    if (i == j) {
        accels[idx][0] = 0.0;
        accels[idx][1] = 0.0;
        accels[idx][2] = 0.0;
        return;
    }

    double dx = pos[i][0] - pos[j][0];
    double dy = pos[i][1] - pos[j][1];
    double dz = pos[i][2] - pos[j][2];

    double magnitude_sq = dx*dx + dy*dy + dz*dz;
    double magnitude    = sqrt(magnitude_sq);

    double accelmag = -1.0 * GRAV_CONSTANT * mass[j] / magnitude_sq;

    double ax = accelmag * dx / magnitude;
    double ay = accelmag * dy / magnitude;
    double az = accelmag * dz / magnitude;

    accels[idx][0] = ax;
    accels[idx][1] = ay;
    accels[idx][2] = az;
}

/*
 * Kernel 2: for each entity i:
 *   - sum accels[i][j] over all j
 *   - update velocity and position exactly like the CPU version:
 *
 *     hVel[i][k] += accel_sum[k] * INTERVAL;
 *     hPos[i][k] += hVel[i][k]   * INTERVAL;
 */
__global__ void updateKernel(vector3 *pos,
                             vector3 *vel,
                             vector3 *accels,
                             int n,
                             double interval)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double ax_sum = 0.0;
    double ay_sum = 0.0;
    double az_sum = 0.0;

    // Sum row i of the accel matrix
    for (int j = 0; j < n; j++) {
        int idx = i * n + j;
        ax_sum += accels[idx][0];
        ay_sum += accels[idx][1];
        az_sum += accels[idx][2];
    }

    // Update velocity
    vel[i][0] += ax_sum * interval;
    vel[i][1] += ay_sum * interval;
    vel[i][2] += az_sum * interval;

    // Update position (using updated velocity)
    pos[i][0] += vel[i][0] * interval;
    pos[i][1] += vel[i][1] * interval;
    pos[i][2] += vel[i][2] * interval;
}

// Track how many steps we've done and how many are required
static int stepsDone  = 0;
static int totalSteps = 0;

#ifdef __cplusplus
extern "C"
#endif
void compute(void){
    int n = NUMENTITIES;

    if (!gpu_initialized) {
        // First time we ever call compute()

        initGpu(n);

        // Compute total number of time steps once.
        // Avoid overflow warnings by using 64-bit for the macro math.
        long long dur = (long long)DURATION;
        long long dt  = (long long)INTERVAL;
        totalSteps = (int)(dur / dt);
        stepsDone  = 0;

        // Copy initial host state to device ONCE
        checkCuda(cudaMemcpy(d_hPos, hPos, n * sizeof(vector3), cudaMemcpyHostToDevice),
                  "cudaMemcpy hPos -> d_hPos (init)");
        checkCuda(cudaMemcpy(d_hVel, hVel, n * sizeof(vector3), cudaMemcpyHostToDevice),
                  "cudaMemcpy hVel -> d_hVel (init)");
        checkCuda(cudaMemcpy(d_mass, mass, n * sizeof(double), cudaMemcpyHostToDevice),
                  "cudaMemcpy mass -> d_mass (init)");
    }

    // --- Perform ONE time-step entirely on the device ---

    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                 (n + blockDim.y - 1) / blockDim.y);

    computeAccelsKernel<<<gridDim, blockDim>>>(d_hPos, d_mass, d_accels, n);
    checkCuda(cudaGetLastError(), "launch computeAccelsKernel");
    checkCuda(cudaDeviceSynchronize(), "sync after computeAccelsKernel");

    int blockSize = 256;
    int gridSize  = (n + blockSize - 1) / blockSize;

    updateKernel<<<gridSize, blockSize>>>(d_hPos, d_hVel, d_accels, n, (double)INTERVAL);
    checkCuda(cudaGetLastError(), "launch updateKernel");
    checkCuda(cudaDeviceSynchronize(), "sync after updateKernel");

    stepsDone++;

    // --- On the FINAL step, copy the result back to the host once ---

    if (stepsDone == totalSteps) {
        checkCuda(cudaMemcpy(hPos, d_hPos, n * sizeof(vector3), cudaMemcpyDeviceToHost),
                  "cudaMemcpy d_hPos -> hPos (final)");
        checkCuda(cudaMemcpy(hVel, d_hVel, n * sizeof(vector3), cudaMemcpyDeviceToHost),
                  "cudaMemcpy d_hVel -> hVel (final)");
    }
}

