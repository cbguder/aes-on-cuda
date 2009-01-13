NVCC=nvcc
NVCCARGS=

all: getDeviceProperties

getDeviceProperties: getDeviceProperties.cu
	$(NVCC) $(NVCCARGS) -o $@ $<

clean:
	$(RM) getDeviceProperties
