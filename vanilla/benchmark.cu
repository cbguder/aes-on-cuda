#include <stdio.h>
#include <time.h>
#include "AES.h"
#include "main.h"

int main(int argc, char **argv) {
	if(argc < 2) {
		printf("USAGE: benchmark FILE\n");
		return 1;
	}

	uint key[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
	uint keySize = 16;

	uint *ct, *pt;
	FILE *f = fopen(argv[1], "rb");
	if(f == NULL) {
		printf("File not found.\n");
		return 1;
	}

	fseek(f, 0, SEEK_END);
	uint ptSize = ftell(f);
	rewind(f);

	if(ptSize % 16 != 0) {
		printf("Plaintext size must be a multiple of AES block size.\n");
		return 1;
	}

	pt = (uint*)malloc(ptSize*sizeof(uint));
	fread(pt, 1, ptSize, f);
	fclose(f);

	ct = (uint *)malloc(ptSize*sizeof(uint));

	clock_t start = clock();
	aes_encrypt_ecb(pt, key, ct, keySize << 3, ptSize >> 4);
	clock_t end = clock();

	printf("%d blocks encrypted in %d/%d seconds.\n", ptSize >> 4, end-start, CLOCKS_PER_SEC);
	
	return EXIT_SUCCESS;
}

uint stringToByteArray(char *str, uint **array) {
	uint i, len  = strlen(str) >> 1;
	*array = (uint *)malloc(len * sizeof(uint));
	
	for(i=0; i<len; i++)
		sscanf(str + i*2, "%02X", *array+i);

	return len;
}

void printHexArray(uint *array, uint size) {
	uint i;
	for(i=0; i<size; i++)
		printf("%02X", array[i]);
	printf("\n");
}
