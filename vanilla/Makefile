NVCC=nvcc

EXECUTABLES=aes benchmark

all: $(EXECUTABLES)

aes: AES.cu AES.h
	$(NVCC) -o $@ $<

benchmark: benchmark.cu AES.h
	$(NVCC) -o $@ $<

clean:
	$(RM) $(EXECUTABLES)
