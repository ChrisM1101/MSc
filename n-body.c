// n-body.c
// This program simulates the motion of n-bodies in a 3D space.
// It uses OpenMP for parallelization.
// The number of threads is set by the environment variable OMP_NUM_THREADS.
// Moustakas Christos, Aristotle University of Thessaloniki 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <unistd.h> // Needed for getopt()

#define G 6.6743e-11 // Gravitational constant
#define NUM_TIMESTEPS 1000 // Number of time steps
#define DT 0.01 // Time step size
#define DIMENSIONS_SIZE 2000 // Size of each dimension of cube space

// Declare N as a variable (not a macro) to allow dynamic changes
int N = 1000; // Number of bodies

// Body structure
typedef struct {
    double x, y, z;    // Position
    double vx, vy, vz; // Velocity
    double mass;       // Mass
} Body;

// Function: Initialize body properties with random values in a 2000m x 2000m x 2000m cube space
void initialize_bodies(Body bodies[N]) {
    for (int i = 0; i < N; i++) {
        bodies[i].x = rand() % (DIMENSIONS_SIZE + 1);
        bodies[i].y = rand() % (DIMENSIONS_SIZE + 1);
        bodies[i].z = rand() % (DIMENSIONS_SIZE + 1);
        bodies[i].vx = bodies[i].vy = bodies[i].vz = 0.0;
        bodies[i].mass = 1e5 + (rand() % 1000); // Mass between 100k and 101k
    }
}

// Compute forces and update velocities
void update_velocities(Body bodies[N]) {
    #pragma omp parallel for shared(bodies)
    for (int i = 0; i < N; i++) {
        double ax = 0.0, ay = 0.0, az = 0.0; // Acceleration
        for (int j = 0; j < N; j++) {
            if (i != j) {
                double dx = bodies[j].x - bodies[i].x;
                double dy = bodies[j].y - bodies[i].y;
                double dz = bodies[j].z - bodies[i].z;
                double dist_sqr = dx * dx + dy * dy + dz * dz + 1e-9; // Avoid division by zero
                double dist = sqrt(dist_sqr); // Distance between bodies i and j
                double force = (G * bodies[i].mass * bodies[j].mass) / dist_sqr; // Force between bodies i and j
                double acceleration = force / bodies[i].mass;
                
                ax += acceleration * (dx / dist);
                ay += acceleration * (dy / dist);
                az += acceleration * (dz / dist);
            }
        }
        // Update velocity
        bodies[i].vx += ax * DT;
        bodies[i].vy += ay * DT;
        bodies[i].vz += az * DT;
    }
}

// Update positions based on velocities
void update_positions(Body bodies[N]) {
    #pragma omp parallel for shared(bodies)
    for (int i = 0; i < N; i++) {
        bodies[i].x += bodies[i].vx * DT;
        bodies[i].y += bodies[i].vy * DT;
        bodies[i].z += bodies[i].vz * DT;
    }
}

int main(int argc, char *argv[]) {

    int opt;

    // Parse command line arguments
    while ((opt = getopt(argc, argv, "n:")) != -1) {
        switch (opt) {
            case 'n':
                N = atol(optarg);
                break;
            default:
                fprintf(stderr, "Usage: %s [-n N]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    // Dynamically allocate bodies
    Body *bodies = (Body *)malloc(N * sizeof(Body));
    if (bodies == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    initialize_bodies(bodies);

    for (int t = 0; t < NUM_TIMESTEPS; t++) {
        update_velocities(bodies);
        update_positions(bodies);
    }

    free(bodies); // Free allocated memory

    return 0;
}