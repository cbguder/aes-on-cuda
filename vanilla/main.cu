#include <stdio.h>
#include "AES.h"
#include "main.h"

int main(int argc, char **argv) {
	if(argc < 3) {
		printf("USAGE: aes KEY PLAINTEXT\n");
		return 1;
	}

	uint ct[16], *key, *pt;
	uint keySize = stringToByteArray(argv[1], &key);
	uint ptSize  = stringToByteArray(argv[2], &pt);

	if(keySize != 16 && keySize != 24 && keySize != 32) {
		printf("Invalid AES key size.\n");
		return 1;
	}

	if(ptSize != 16) {
		printf("Invalid AES block size.\n");
		return 1;
	}

	aes_encrypt(pt, key, ct, keySize << 3);

	printHexArray(ct, 16);

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
