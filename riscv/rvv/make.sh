~/toolchain/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1/bin/riscv64-unknown-linux-gnu-as -o test.o test.S -march=rv64imafdv0p7

~/toolchain/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1/bin/riscv64-unknown-linux-gnu-gcc -c gflops_benchmark.c -march=rv64imafdv0p7

~/toolchain/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1/bin/riscv64-unknown-linux-gnu-gcc -o gflops_benchmark gflops_benchmark.o test.o -march=rv64imafdv0p7
