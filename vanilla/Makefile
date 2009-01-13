NVCC=nvcc
NVCCARGS=-g -deviceemu

EXECUTABLES=aes getDeviceProperties

all: $(EXECUTABLES)

aes: AES.cu AES.h
	$(NVCC) $(NVCCARGS) -o $@ $<

getDeviceProperties: getDeviceProperties.cu
	$(NVCC) $(NVCCARGS) -o $@ $<

clean:
	$(RM) $(EXECUTABLES) 
