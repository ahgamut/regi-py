
CC := gcc
CXX := g++

all: main
	./main

main: main.o card.o deck.o
	$(CXX) -o $@ $^

%.o: %.cc
	$(CXX) -c -o $@ $< -I. -DUSE_UNICODE

clean:
	rm *.o
