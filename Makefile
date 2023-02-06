nn.exe: main.o neuralNetwork.o
	g++ -o nn.exe main.o neuralNetwork.o
main.o:main.cpp neuralNetwork.h
	g++ -c main.cpp
neuralNetwork.o:neuralNetwork.cpp neuralNetwork.h
	g++ -c neuralNetwork.cpp
clean:
	rm *.out *.o *.stackdump *~
