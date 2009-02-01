#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "AES.h"
#include "main.h"

using namespace std;

int main(int argc, char **argv) {
	if(argc < 2) {
		printf("USAGE: benchmark FILE\n");
		return 1;
	}

	byte key[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
	uint keySize = 16;

	uint *ct, *pt;
	
	FILE *f = fopen(argv[1], "rb");
	if(f == NULL) {
		printf("File not found.\n");
		return 1;
	}

	fseek(f, 0, SEEK_END);
	uint f_size = ftell(f);
	rewind(f);

	if(f_size % 4*sizeof(uint) != 0) {
		printf("Plaintext size must be a multiple of AES block size.\n");
		return 1;
	}

	uint ptSize = f_size / sizeof(uint);

#ifdef ASYNC
	cudaMallocHost((void**)&pt, f_size);
	cudaMallocHost((void**)&ct, f_size);
#else
	pt = (uint*)malloc(f_size);
	ct = (uint *)malloc(f_size);
#endif

	fread(pt, sizeof(uint), ptSize, f);
	fclose(f);

	AES *aes = new AES();
	aes->makeKey(key, keySize << 3, DIR_ENCRYPT);

	clock_t start = clock();

#ifdef ASYNC
	aes->encrypt_ecb_async(pt, ct, ptSize >> 2);
#else
	aes->encrypt_ecb(pt, ct, ptSize >> 2);
#endif

	clock_t end = clock();

	printf("%d blocks encrypted in %d/%d seconds.\n", ptSize >> 2, end-start, CLOCKS_PER_SEC);

	return 0;
}

uint stringToByteArray(char *str, byte **array) {
	uint i, len  = strlen(str) >> 1;
	*array = (byte *)malloc(len * sizeof(byte));
	
	for(i=0; i<len; i++)
		sscanf(str + i*2, "%02X", *array+i);

	return len;
}

uint stringToByteArray(char *str, uint **array) {
	uint i, len  = strlen(str) >> 3;
	*array = (uint *)malloc(len * sizeof(uint));
	
	for(i=0; i<len; i++)
		sscanf(str + i*8, "%08X", *array+i);

	return len;
}

void printHexArray(uint *array, uint size) {
	uint i;
	for(i=0; i<size; i++)
		printf("%08X", array[i]);
	printf("\n");
}
