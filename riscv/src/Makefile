CXX = ~/toolchain/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1/bin/riscv64-unknown-linux-gnu-g++
CC = ~/toolchain/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1/bin/riscv64-unknown-linux-gnu-gcc
#AR = ar cqs

TOP_DIR = .

TARGET_EXE ?= unit_test

#CXX_SOURCE += $(TOP_DIR)/src/unit_test_conv.cpp
CXX_SOURCE += ./test_matrix_multiply.cpp

####################head file###################
DEP_INC_DIR += -I$(TOP_DIR)/
#DEP_INC_DIR += -I$(TOP_DIR)/openblas-c910/include

OBJ_CPP := $(patsubst %.cpp, %.o, $(CXX_SOURCE))
OBJ_C := $(patsubst %.c, %.o, $(C_SOURCE))

DEFINE_MACRO := 

CPPFLAGS := -Wall -fpermissive -std=c++11 -march=rv64imafdv0p7 
CPPFLAGS += -O2 
#CPPFLAGS += -Og -g
CPPFLAGS += $(DEP_INC_DIR) $(DEFINE_MACRO)

#CPPFLAGS += -ftree-vectorize

CFLAGS += $(DEP_INC_DIR) $(DEFINE_MACRO)
LDFLAGS += -static

#LDFLAGS += -lopenblas -L./openblas-c910/lib/ -lpthread

$(TARGET_EXE):$(OBJ_CPP)$(OBJ_C)
	$(CXX) -o $@ $^ $(LDFLAGS)
	sshpass -p "licheepi" scp $(TARGET_EXE) sipeed@192.168.0.156:/home/sipeed/test/gflops/

%.o : %.cpp
	$(CXX) $(CPPFLAGS) -o $@ -c $< 

%.o : %.c
	$(CC) $(CFLAGS) -o $@ -c $< 

clean:
	rm -rf $(OBJ_CPP) $(OBJ_C) $(TARGET_EXE)
