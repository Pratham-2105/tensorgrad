CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -O2 -g
HEADERS  := matrix.hpp value.hpp mnist.hpp serialize.hpp

tensorgrad: main.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o tensorgrad main.cpp

run: tensorgrad
	./tensorgrad

predict: predict.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o predict predict.cpp
	./predict

bigram: bigram.cpp makemore.hpp
	$(CXX) $(CXXFLAGS) -o bigram bigram.cpp
	./bigram

clean:
	rm -f tensorgrad predict bigram

.PHONY: run predict bigram clean
