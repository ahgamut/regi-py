AR := ar
CC := gcc
CXX := g++

CXXFLAGS  :=  -I. -DUSE_UNICODE -DNUM_PLAYERS=4\
			 -std=c++17 -O3 -c -g3 -Wall -Wpedantic -Wextra \
			 -fno-omit-frame-pointer -Wno-sign-compare
LINKFLAGS := -fno-omit-frame-pointer
VALFLAGS := --tool=memcheck --leak-check=full --leak-resolution=high --show-leak-kinds=all --errors-for-leak-kinds=all

all: main
	./main

dbg: main
	valgrind $(VALFLAGS) ./main

main: main.o libregi.a
	$(CXX) $(LINKFLAGS) -o $@ $^

libregi.a: card.o deck.o \
	player.o enemy.o \
	regi.o combo.o effects.o \
	interact.o dfsel.o \
	console.o
	$(AR) rcs $@ $^

%.o: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f *.o libregi.a main
