all: compile link run clean

build: compile link

compile:
	g++ -c *.cpp
link:
	g++ -o main.out *.o -larmadillo
run:
	./main.out 2
	./main.out 3
	./main.out 4
	./main.out 5
clean:
	rm *.o *.out
