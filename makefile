CC = nvcc
CFLAGS = -lpng --extended-lambda

all: domain_color

domain_color: domain_color.cu
	$(CC) $(CFLAGS) -o bin/domain_color domain_color.cu

clean:
	rm -f bin/domain_color