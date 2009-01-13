NVCC=nvcc
NVCCARGS=

all: aes

aes: AES.cu AES.h
	$(NVCC) $(NVCCARGS) -o $@ $<

clean:
	$(RM) aes
