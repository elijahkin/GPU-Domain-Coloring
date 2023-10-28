CC = nvcc
CFLAGS = -lpng --extended-lambda

all: complex_plot

complex_plot: complex_plot.cu
	$(CC) $(CFLAGS) -o bin/complex_plot complex_plot.cu

clean:
	rm -f bin/complex_plot