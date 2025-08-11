
CC := gcc
CXX := g++

all: main
	./main

main: main.o \
	card.o deck.o \
	player.o enemy.o \
	regi.o combo.o effects.o interact.o
	$(CXX) -o $@ $^

%.o: %.cc
	$(CXX) -c -o $@ $< -I. -DUSE_UNICODE

clean:
	rm -f *.o main
