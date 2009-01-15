NVCC=nvcc

EXECUTABLES=aes aes_ecb benchmark benchmark_cpb
OBJECTS=AES.o AES_cpb.o benchmark.o main.o main_ecb.o

all: $(EXECUTABLES)

aes: AES.o main.o
	$(NVCC) -o $@ $^

aes_ecb: AES.o main_ecb.o
	$(NVCC) -o $@ $^

benchmark: AES.o benchmark.o
	$(NVCC) -o $@ $^

benchmark_cpb: AES_cpb.o benchmark.o
	$(NVCC) -o $@ $^
	
main.o: main.cu main.h
	$(NVCC) -c -o $@ $<

main_ecb.o: main_ecb.cu main.h
	$(NVCC) -c -o $@ $<

benchmark.o: benchmark.cu main.h
	$(NVCC) -c -o $@ $<

AES.o: AES.cu AES.h
	$(NVCC) -c -o $@ $<

AES_cpb.o: AES.cu AES.h
	$(NVCC) -DNO_COPYBACK -c -o $@ $<

clean:
	$(RM) $(EXECUTABLES) $(OBJECTS)
