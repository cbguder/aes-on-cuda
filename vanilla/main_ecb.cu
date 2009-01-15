#include <stdio.h>
#include "AES.h"
#include "main.h"

int main(int argc, char **argv) {
	if(argc < 3) {
		printf("USAGE: aes_ecb KEY PLAINTEXT [PLAINTEXT...]\n");
		return 1;
	}

	uint *key, *ct, *pt;
	uint keySize = stringToByteArray(argv[1], &key);
	uint ptSize  = stringToByteArray(argv[2], &pt);

	if(keySize != 16 && keySize != 24 && keySize != 32) {
		printf("Invalid AES key size.\n");
		return 1;
	}

	if(ptSize % 16 != 0) {
		printf("Plaintext size must be a multiple of AES block size.\n");
		return 1;
	}

	ct = (uint *)malloc(ptSize*sizeof(uint));

	aes_encrypt_ecb(pt, key, ct, keySize << 3, ptSize >> 4);

	printHexArray(ct, ptSize);

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
