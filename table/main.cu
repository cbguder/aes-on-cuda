#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "AES.h"
#include "main.h"

using namespace std;

int main(int argc, char **argv) {
	if(argc < 3) {
		printf("USAGE: aes KEY PLAINTEXT\n");
		return 1;
	}

	byte *key;
	uint ct[16], *pt;
	uint keySize = stringToByteArray(argv[1], &key);
	uint ptSize  = stringToByteArray(argv[2], &pt);

	if(keySize != 16 && keySize != 24 && keySize != 32) {
		printf("Invalid AES key size.\n");
		return 1;
	}

	if(ptSize != 4) {
		printf("Invalid AES block size.\n");
		return 1;
	}

	copyTables();

	AES *aes = new AES();
	aes->makeKey(key, keySize << 3, DIR_ENCRYPT);
	aes->encrypt(pt, ct);

	printHexArray(ct, 4);

	freeTables();

	return 0;
}

void copyTables() {
	int tableSize = 256*sizeof(uint);

	cudaMalloc((void**)&cTe0, sizeof(Te0));
	cudaMalloc((void**)&cTe1, sizeof(Te1));
	cudaMalloc((void**)&cTe2, sizeof(Te2));
	cudaMalloc((void**)&cTe3, sizeof(Te3));
	cudaMalloc((void**)&cTe4, sizeof(Te4));

	cudaMalloc((void**)&cTd0, sizeof(Td0));
	cudaMalloc((void**)&cTd1, sizeof(Td1));
	cudaMalloc((void**)&cTd2, sizeof(Td2));
	cudaMalloc((void**)&cTd3, sizeof(Td3));
	cudaMalloc((void**)&cTd4, sizeof(Td4));

	cudaMemcpy(cTe0, Te0, tableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(cTe1, Te1, tableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(cTe2, Te2, tableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(cTe3, Te3, tableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(cTe4, Te4, tableSize, cudaMemcpyHostToDevice);

	cudaMemcpy(cTd0, Td0, tableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(cTd1, Td1, tableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(cTd2, Td2, tableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(cTd3, Td3, tableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(cTd4, Td4, tableSize, cudaMemcpyHostToDevice);
}

void freeTables() {
	cudaFree(cTe0);
	cudaFree(cTe1);
	cudaFree(cTe2);
	cudaFree(cTe3);
	cudaFree(cTe4);

	cudaFree(cTd0);
	cudaFree(cTd1);
	cudaFree(cTd2);
	cudaFree(cTd3);
	cudaFree(cTd4);
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
