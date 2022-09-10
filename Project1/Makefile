
all: compile link test run clean

compile:
	g++ -c ./src/solvers.cpp -I include

	g++ -c main.cpp -I include
link:
	g++ -o main.out main.o solvers.o -larmadillo
test:
	g++ -c ./tests/test_solvers.cpp -I include
	g++ -o test_solvers.out test_solvers.o solvers.o -larmadillo
	./test_solvers.out

run:
	./main.out
	python3 ./scripts/plot_approximations.py
	python3 ./scripts/plot_absolute_errors.py
	python3 ./scripts/plot_relative_errors.py
clean:
	rm *.out *.o
