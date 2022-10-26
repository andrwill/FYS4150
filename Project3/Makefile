all: compile link run clean
compile:
	g++ -c src/Particle.cpp -I include
	g++ -c src/PenningTrap.cpp -I include
	g++ -c main.cpp -I include
link:
	g++ -o main.out main.o Particle.o PenningTrap.o -larmadillo
run:
	./main.out
clean:
	rm main.out *.o
