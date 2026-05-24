CXX = g++

CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -fopenmp

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

TARGET = $(MAKEFILE_DIR)/mmas_core

SRCS = $(MAKEFILE_DIR)/main.cpp $(MAKEFILE_DIR)/aco.cpp

OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
