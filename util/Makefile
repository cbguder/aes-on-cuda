NVCC=nvcc
NVCCARGS=

getDeviceProperties: getDeviceProperties.cu
	$(NVCC) $(NVCCARGS) -o $@ $<

clean:
	$(RM) getDeviceProperties
