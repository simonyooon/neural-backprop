#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <iostream>
#include <fstream>

#include <cmath>

#include <vector>
#include <string>

using namespace std;

class neuralNetwork{
    public:
        neuralNetwork(ifstream &initfile);
        int train(ifstream &trainfile, double learnRate, int epochs);
        int test(ifstream &testfile, ofstream &outfile);
        void save(ostream &outfile);
    private:
        class link;
        class neuron{
            public:
                double inputValue;
                double activation;
                double error;
                vector<link> inLinks;
                vector<link> outLinks;
        };
        class link{
            public:
                double weight;
                neuron *connectedNeuron;
        };
    
        
        class trainingSets{
        public:
            vector<double> inputs;
            vector<int> outputs;
        };
        
        int numLayers;
        vector<int> layerSize;
        vector<vector<neuron> > layers;
        double sigmoid(double inputValue);
        double sigmoidPrime(double inputValue);
};

#endif 
