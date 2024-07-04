AS := ~/toolchain/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1/bin/riscv64-unknown-linux-gnu-as
GCC := ~/toolchain/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1/bin/riscv64-unknown-linux-gnu-gcc

CFLAGS := -march=rv64imafdv0p7

clean:
	rm gflops_benchmark.o gbench test.o

gbench:
	$(AS) -o test.o test.S $(CFLAGS)
	$(GCC) -c gflops_benchmark.c $(CFLAGS)
	$(GCC) -o $@ test.o gflops_benchmark.o $(CFLAGS)


scp: gbench
	sshpass -p "licheepi" scp gbench sipeed@192.168.0.156:/home/sipeed/test/gflops/

