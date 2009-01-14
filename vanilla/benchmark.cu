#include <stdio.h>
#include <time.h>

#include "AES.h"

#define GET(M,X,Y) ((M)[((Y) << 2) + (X)])

const uint size = 4*4*sizeof(uint);

int main(int argc, char **argv) {
	if(argc < 3) {
		printf("USAGE: aes KEY PLAINTEXT\n");
		return 1;
	}

	uint key[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
	uint keySize = 16;

	uint ct[16], *pt;
	uint ptSize  = stringToByteArray(argv[2], &pt);

	if(ptSize != 16) {
		printf("Invalid AES block size.\n");
		return 1;
	}

	clock_t start = clock();
	aes_encrypt(pt, key, ct, keySize << 3);
	clock_t end = clock();

	printf("1 block encrypted in %d/%d seconds.\n", end-start, CLOCKS_PER_SEC);

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

void aes_encrypt(uint *pt, uint *key, uint *ct, uint keysize) {
	uint i, *cp, *W, *cW, Nk, Nr;
	Nk = keysize >> 5;
	Nr = Nk + 6;

	uint s = ((Nr+1) * sizeof(uint)) << 4;
	W = (uint *)malloc(s);
	cudaMalloc((void**)&cW, s);
	ExpandKeys(key, keysize, W, Nk, Nr);
	cudaMemcpy(cW, W, s, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cp, size);
	cudaMemcpy(cp, pt, size, cudaMemcpyHostToDevice);

	AddRoundKey<<<1,16>>>(cp, cW);
	for(i=1; i<Nr; i++) {
		SubBytes<<<1,16>>>(cp);
		ShiftRows<<<1,4>>>(cp);
		MixColumns<<<1,4>>>(cp);
		AddRoundKey<<<1,16>>>(cp, cW+(i << 4));
	}
	SubBytes<<<1,16>>>(cp);
	ShiftRows<<<1,4>>>(cp);
	AddRoundKey<<<1,16>>>(cp, cW+(i << 4));

	cudaMemcpy(ct, cp, size, cudaMemcpyDeviceToHost);
}

void ExpandKeys(uint *key, uint keysize, uint *W, uint Nk, uint Nr) {
	uint i, j, cols, temp, tmp[4];
	cols = (Nr + 1) << 2;

	memcpy(W, key, (keysize >> 3)*sizeof(uint));

	for(i=Nk; i<cols; i++) {
		for(j=0; j<4; j++)
			tmp[j] = GET(W, j, i-1);
		if(Nk > 6) {
			if(i % Nk == 0) {
				temp   = hsbox[tmp[0]] ^  (Rcon[i/Nk] & 0x000000ff);
				tmp[0] = hsbox[tmp[1]] ^ ((Rcon[i/Nk] & 0xff000000) >> 24);
				tmp[1] = hsbox[tmp[2]] ^ ((Rcon[i/Nk] & 0x00ff0000) >> 16);
				tmp[2] = hsbox[tmp[3]] ^ ((Rcon[i/Nk] & 0x0000ff00) >>  8);
				tmp[3] = temp;
			} else if(i % Nk == 4) {
				tmp[0] = hsbox[tmp[0]];
				tmp[1] = hsbox[tmp[1]];
				tmp[2] = hsbox[tmp[2]];
				tmp[3] = hsbox[tmp[3]];
			}
		} else {
			if(i % Nk == 0) {
				temp   = hsbox[tmp[0]] ^  (Rcon[i/Nk] & 0x000000ff);
				tmp[0] = hsbox[tmp[1]] ^ ((Rcon[i/Nk] & 0xff000000) >> 24);
				tmp[1] = hsbox[tmp[2]] ^ ((Rcon[i/Nk] & 0x00ff0000) >> 16);
				tmp[2] = hsbox[tmp[3]] ^ ((Rcon[i/Nk] & 0x0000ff00) >>  8);
				tmp[3] = temp;
			}
		}
		for(j=0; j<4; j++)
			GET(W, j, i) = GET(W, j, i-Nk) ^ tmp[j];
	}
}

__global__ void SubBytes(uint *state) {
	uint i = threadIdx.x;
	state[i] = sbox[state[i]];
}

__global__ void ShiftRows(uint *state) {
	uint row  = threadIdx.x;
	uint i, tmp[4];

	for(i=0; i<4; i++)
		tmp[i] = state[row + 4*(i+row) % 16];
	for(i=0; i<4; i++)
		state[row + 4*i] = tmp[i];
}

#define xtime(x) ((x<<1) ^ (((x>>7) & 1) * 0x1b))
__global__ void MixColumns(uint *state) {
	uint col  = threadIdx.x;
	uint base = col << 2;
	uint t, Tmp, Tm;

	t   = state[base];
	Tmp = state[base] ^ state[base + 1] ^ state[base + 2] ^ state[base + 3];
	Tm  = state[base    ] ^ state[base + 1]; Tm = xtime(Tm) & 0xff; state[base    ] ^= Tm ^ Tmp;
	Tm  = state[base + 1] ^ state[base + 2]; Tm = xtime(Tm) & 0xff; state[base + 1] ^= Tm ^ Tmp;
	Tm  = state[base + 2] ^ state[base + 3]; Tm = xtime(Tm) & 0xff; state[base + 2] ^= Tm ^ Tmp;
	Tm  = state[base + 3] ^ t;               Tm = xtime(Tm) & 0xff; state[base + 3] ^= Tm ^ Tmp;
}

__device__ void AddRoundKey(uint *state, uint *key) {
	uint i = threadIdx.x;
	state[i] ^= key[i];
}
