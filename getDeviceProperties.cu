#include <stdio.h>

int main(int argc, char **argv) {
	struct cudaDeviceProp prop;

	cudaGetDeviceProperties(&prop, 0);

	printf("name:                %s\n", prop.name);
	printf("totalGlobalMem:      %zd\n", prop.totalGlobalMem);
	printf("sharedMemPerBlock:   %zd\n", prop.sharedMemPerBlock);
	printf("regsPerBlock:        %d\n", prop.regsPerBlock);
	printf("warpSize:            %d\n", prop.warpSize);
	printf("memPitch:            %zd\n", prop.memPitch);
	printf("maxThreadsPerBlock:  %d\n", prop.maxThreadsPerBlock);
	printf("maxThreadsDim:       %dx%dx%d\n", prop.maxThreadsDim[0],
	                                          prop.maxThreadsDim[1],
	                                          prop.maxThreadsDim[2]);
	printf("maxGridSize:         %dx%dx%d\n", prop.maxGridSize[0],
	                                          prop.maxGridSize[1],
	                                          prop.maxGridSize[2]);
	printf("totalConstMem:       %zd\n", prop.totalConstMem);
	printf("major:               %d\n", prop.major);
	printf("minor:               %d\n", prop.minor);
	printf("clockRate:           %d\n", prop.clockRate);
	printf("textureAlignment:    %zd\n", prop.textureAlignment);
	printf("deviceOverlap:       %d\n", prop.deviceOverlap);
	printf("multiProcessorCount: %d\n", prop.multiProcessorCount);

	return EXIT_SUCCESS;
}
