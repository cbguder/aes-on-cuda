NVCC=nvcc

EXECUTABLES=aes aes_ecb benchmark benchmark_async benchmark_con benchmark_cpb benchmark_con_cpb
OBJECTS=AES.o AES_benchmark.o AES_benchmark_con.o AES_benchmark_con_cpb.o AES_benchmark_cpb.o benchmark.o benchmark_async.o main.o main_ecb.o
AES_FILES=AES.cu AES.h AES.tab BlockCipher.h

all: $(EXECUTABLES)

aes: AES.o main.o
	$(NVCC) -o $@ $^

aes_ecb: AES.o main_ecb.o
	$(NVCC) -o $@ $^

benchmark: AES_benchmark.o benchmark.o
	$(NVCC) -o $@ $^

benchmark_async: AES_benchmark.o benchmark_async.o
	$(NVCC) -o $@ $^

benchmark_con: AES_benchmark_con.o benchmark.o
	$(NVCC) -o $@ $^

benchmark_cpb: AES_benchmark_cpb.o benchmark.o
	$(NVCC) -o $@ $^

benchmark_con_cpb: AES_benchmark_con_cpb.o benchmark.o
	$(NVCC) -o $@ $^

AES.o: $(AES_FILES)
	$(NVCC) -c -o $@ $<

AES_benchmark.o: $(AES_FILES)
	$(NVCC) -DBENCHMARK -c -o $@ $<

AES_benchmark_con.o: $(AES_FILES)
	$(NVCC) -DBENCHMARK -DUSE_CONSTANT -c -o $@ $<

AES_benchmark_cpb.o: $(AES_FILES)
	$(NVCC) -DBENCHMARK -DNO_COPYBACK -c -o $@ $<

AES_benchmark_con_cpb.o: $(AES_FILES)
	$(NVCC) -DBENCHMARK -DUSE_CONSTANT -DNO_COPYBACK -c -o $@ $<

main.o: main.cu main.h
	$(NVCC) -c -o $@ $<

main_ecb.o: main_ecb.cu main.h
	$(NVCC) -c -o $@ $<

benchmark.o: benchmark.cu main.h
	$(NVCC) -c -o $@ $<

benchmark_async.o: benchmark.cu main.h
	$(NVCC) -DASYNC -c -o $@ $<

clean:
	$(RM) $(EXECUTABLES) $(OBJECTS)
