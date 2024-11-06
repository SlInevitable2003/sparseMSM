cpp_standard := -std=c++17

test: alt_bn128.cpp bigint.cpp test.cpp
	g++ -o test test.cpp bigint.cpp alt_bn128.cpp $(cpp_standard)

clean:
	rm -f test