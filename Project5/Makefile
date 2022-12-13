all: compile link run clean

compile:
	g++ -c main.cpp -larmadillo -fopenmp
link:
	g++ -o main.out main.o -larmadillo -fopenmp
run:
	./main.out
clean:
	rm *.out *.o
