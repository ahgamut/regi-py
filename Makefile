CXXFLAGS  =  -I. -DUSE_UNICODE -DNUM_PLAYERS=4\
			 -std=c++11 -O3 -c -g3 -Wall -Wpedantic -Wextra \
			 -fno-omit-frame-pointer -Wno-sign-compare
LINKFLAGS = -fno-omit-frame-pointer
CC := gcc
CXX := g++
VALFLAGS = --tool=memcheck --leak-check=full --leak-resolution=high --show-leak-kinds=all --errors-for-leak-kinds=all

dbg: main
	valgrind $(VALFLAGS) ./main

all: main
	./main

main: main.o \
	card.o deck.o \
	player.o enemy.o \
	regi.o combo.o effects.o \
	interact.o dfsel.o \
	console.o
	$(CXX) $(LINKFLAGS) -o $@ $^

%.o: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f *.o main
