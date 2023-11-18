CC = nvcc
CFLAGS = --extended-lambda

all: complex_plot

complex_plot: src/renders.cu
	$(CC) $(CFLAGS) -o bin/complex_plot src/renders.cu

clean:
	rm -f bin/complex_plot