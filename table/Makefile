NVCC=nvcc

all: aes

aes: AES.o main.o
	$(NVCC) -o aes $^

AES.o: AES.cu AES.h BlockCipher.h
	$(NVCC) -c -o $@ $<

main.o: main.cu main.h
	$(NVCC) -c -o $@ $<

clean:
	$(RM) aes AES.o main.o
