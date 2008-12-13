NVCC=nvcc
NVCCARGS=-g -deviceemu

aes: AES.cu AES.h
	$(NVCC) $(NVCCARGS) -o $@ $<

clean:
	$(RM) aes
