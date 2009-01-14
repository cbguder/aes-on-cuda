EXECUTABLES=aes aes_ecb benchmark

all: $(EXECUTABLES)

aes: AES.o main.o
	$(CXX) -o $@ $^

aes_ecb: AES.o main_ecb.o
	$(CXX) -o $@ $^

benchmark: AES.o benchmark.o
	$(CXX) -o $@ $^

clean:
	$(RM) $(EXECUTABLES) AES.o main.o main_ecb.o
