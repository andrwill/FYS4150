all:
	g++ -c main.cpp -I include -fopenmp
	g++ -o main.out main.o -larmadillo -fopenmp
	./main.out
	rm *.out *.o
