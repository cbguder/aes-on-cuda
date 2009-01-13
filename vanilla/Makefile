NVCC=nvcc
NVCCARGS=

aes: AES.cu AES.h
	$(NVCC) $(NVCCARGS) -o $@ $<

clean:
	$(RM) aes
