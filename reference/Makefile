all: aes aes_ecb

aes: AES.o main.o
	$(CXX) -o $@ $^

aes_ecb: AES.o main_ecb.o
	$(CXX) -o $@ $^

clean:
	$(RM) aes aes_ecb AES.o main.o main_ecb.o
