CC = nvcc
CFLAGS = --extended-lambda

all: stills window

stills:
	$(CC) $(CFLAGS) -o bin/stills src/stills.cu

window:
	$(CC) $(CFLAGS) -o bin/window src/window.cu -lGL -lglut

clean:
	rm -f bin/stills
	rm -f bin/window
