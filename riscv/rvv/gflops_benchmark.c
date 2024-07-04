// Hisi 3519A gFLOPS
#include <time.h>
#include <stdio.h>

#define LOOP (1e9)
#define OP_FLOATS (240)

void TEST(int);

extern void TEST_ADD(int);
extern void TEST_S(int);

static double get_time(struct timespec *start,
                       struct timespec *end) {
    return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec) * 1e-9;
}



int main() {
    struct timespec start, end;
    double time_used = 0.0;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    TEST(LOOP);
//    TEST_ADD(LOOP);
//    TEST_S(LOOP);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    time_used = get_time(&start, &end);

    printf("time_used = %f \r\n",time_used);

    printf("perf: %.6lf \r\n", LOOP * OP_FLOATS * 1.0 * 1e-9 / time_used);
//    printf("perf: %.6lf \r\n", LOOP * 31 * 1.0 * 1e-9 / time_used);
}
