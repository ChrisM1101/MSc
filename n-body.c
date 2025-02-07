// n-body.c
// This program simulates the motion of n-bodies in a 2D space.
// It uses OpenMP for parallelization.
// The number of threads is set by the environment variable OMP_NUM_THREADS.
// Moustakas Christos, Aristotle University of Thessaloniki 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define G 6.6743e-11 // Gravitational constant
#define N 1000 // Number of bodies
#define NUM_TIMESTEPS 1000 // Number of time steps
#define DT 0.01 // Time step size