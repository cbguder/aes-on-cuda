NVCC=nvcc

EXECUTABLES=aes aes_ecb benchmark

all: $(EXECUTABLES)

aes: AES.o main.o
	$(NVCC) -o $@ $^

aes_ecb: AES.o main_ecb.o
	$(NVCC) -o $@ $^

benchmark: AES_benchmark.o benchmark.o
	$(NVCC) -o $@ $^

AES.o: AES.cu AES.h AES.tab BlockCipher.h
	$(NVCC) -c -o $@ $<

AES_benchmark.o: AES_benchmark.cu AES.h AES.tab BlockCipher.h
	$(NVCC) -c -o $@ $<

main.o: main.cu main.h
	$(NVCC) -c -o $@ $<

main_ecb.o: main_ecb.cu main.h
	$(NVCC) -c -o $@ $<

benchmark.o: benchmark.cu main.h
	$(NVCC) -c -o $@ $<

clean:
	$(RM) $(EXECUTABLES) AES.o AES_benchmark.o main.o
