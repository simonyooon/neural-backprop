
#include "neuralNetwork.h"
#include <iomanip>
using namespace std;

double A ,B, C, D;
double overall, precision, recall, f1; // metrics
double avgOverall, avgPrecision, avgRecall, avgF1;

neuralNetwork::neuralNetwork(ifstream &initfile){
    this -> numLayers = 3; // layer count                        
    this -> layerSize.resize(this -> numLayers);
    this -> layers.resize(this -> numLayers);
    
    for (int i = 0;i < numLayers; i++){
        initfile >> this -> layerSize[i];
        this -> layerSize[i] ++; // increment to incorporate bias input
        this -> layers[i].resize(this -> layerSize[i]);
    }
    // w0 bias is set to -1
    for (int i = 0; i < this -> numLayers; i++){
        this -> layers[i][0].activation = -1;
    }
    // weight config
    for (int i = 0; i< this -> numLayers - 1; i ++){
        for (int j = 1; j < this -> layerSize[i + 1]; j++){
            for (int k = 0; k < this -> layerSize[i]; k++){
                double weight;
                initfile >> weight;
                link inLinks, outLinks;
                outLinks.weight = weight;
                outLinks.connectedNeuron = &this -> layers[i + 1][j];
                this -> layers[i][k].outLinks.push_back(outLinks);
                
                inLinks.weight = weight;
                inLinks.connectedNeuron = &this -> layers[i][k];
                this -> layers[i+1][j].inLinks.push_back(inLinks);
            }
        }
    }
}

int neuralNetwork::train(ifstream &trainfile, double learnRate,  int epoch){
    vector<trainingSets> trainingSet;
    int inputN, outputN, setN;
  
    trainfile >> setN >> inputN >> outputN;
    trainingSet.resize(setN);
    for (int i = 0; i < setN; i++){
        trainingSet[i].inputs.resize(inputN);
        trainingSet[i].outputs.resize(outputN);
        for(int j = 0; j < inputN; j++){
            trainfile >> trainingSet[i].inputs[j];
        }
        for(int k = 0; k < outputN; k++){
            trainfile >> trainingSet[i].outputs[k];
        }
    }
    
    // backprop algo
    int outputLayerI = this -> numLayers -1;
    for (int i = 0; i < epoch; i++){
        for (int j = 0; j < setN; j++){
            for (int k = 0; k < inputN; k++){
                // copying input vector of single training samples to inputs nodes of the network
                this -> layers[0][k+1].activation = trainingSet[j].inputs[k];
            }
            for (int m = 1; m < this -> numLayers; m++){
                for (int n = 1; n < this -> layerSize[m]; n++){
                    this -> layers[m][n].inputValue = 0;
                    vector<link>::iterator it;
                    for (it = this -> layers[m][n].inLinks.begin(); it != this -> layers[m][n].inLinks.end(); it++){
                        this -> layers[m][n].inputValue += it->weight * it->connectedNeuron->activation;
                    }
                    this ->layers[m][n].activation = this -> sigmoid(this -> layers[m][n].inputValue);
                }
            }
            // error signal from output to input layer
            for (int p = 1; p < this ->layerSize[outputLayerI]; p++ ){
                this -> layers[outputLayerI][p].error = this -> sigmoidPrime(this -> layers[outputLayerI][p]. inputValue) * (trainingSet[j].outputs[p -1] -  this -> layers[outputLayerI][p].activation);
            }
            for (int x = outputLayerI -1; x > 0; x--){
                for (int y = 1; y < this -> layerSize[x]; y ++){
                    double sum = 0;
                    vector<link>::iterator it;
                    for (it = this -> layers[x][y].outLinks.begin();it!= this -> layers[x][y].outLinks.end(); it++){
                        sum += it -> weight * it -> connectedNeuron -> error;
                    }
                    this -> layers[x][y].error = this -> sigmoidPrime(this -> layers[x][y].inputValue) * sum;
                }
            }
            for (int p = 1; p < this -> numLayers; p++){
                for (int q = 1; q < this -> layerSize[p]; q++){
                    vector<link>::iterator it;
                    for (it = this -> layers[p][q].inLinks.begin(); it != this -> layers[p][q].inLinks.end(); it ++){
                        // Update weights in both directions
                        it -> weight = it -> weight + learnRate * it -> connectedNeuron -> activation * this ->layers[p][q].error;
                        it -> connectedNeuron -> outLinks[q -1].weight = it -> weight;
                    }
                }
            }
            
        }
    }
        return 0;
}

int neuralNetwork::test(ifstream &testfile, ofstream &outfile){
    int setN, inputN, outputN;
    vector<trainingSets> example;
    vector<vector<double> > result;

    testfile >> setN >> inputN >> outputN;
    example.resize(setN);
    result.resize(outputN);
    
    for(int i = 0; i < setN; i++){
        example[i].inputs.resize(inputN);
         example[i].outputs.resize(outputN);
        for (int j = 0; j < inputN; j++){
            testfile >> example[i].inputs[j];
        }
        for (int k = 0; k < outputN; k++ ){
            testfile >> example[i].outputs[k];

            //single cases
            if (i == 0){
                result[k].resize(4);
                for (int m = 0; m < 4; m++){
                    result[k][m] = 0;
                }
            }
        }
    }
    
    int outputLayerI = this -> numLayers -1;
    for (int c = 0; c < setN; c++){
        for (int i = 0; i < inputN; i++){
            this -> layers[0][i+1].activation = example[c].inputs[i];
        }
        for (int l = 1; l < this -> numLayers; l++){
            for (int j = 1; j < this -> layerSize[l]; j ++ ){
                this -> layers[l][j].inputValue = 0;
                vector<link>::iterator it;
                for (it = this -> layers[l][j]. inLinks.begin(); it != this -> layers[l][j].inLinks.end(); it ++){
                    this -> layers[l][j].inputValue += it -> weight * (it -> connectedNeuron-> activation);
                }
                this -> layers[l][j].activation = this -> sigmoid(this -> layers[l][j].inputValue);
            }
        }

        // regularization
        for (int n = 1; n < this -> layerSize[outputLayerI]; n ++ ){
            if (this -> layers[outputLayerI][n].activation >= 0.5){
                if (example[c].outputs[n-1]){
                    result[n-1][0]++;
                }
                else {
                    result[n-1][1]++;
                }
            }
            else {
                if (example[c].outputs[n-1]){
                    result[n-1][2]++;
                }
                else {
                    result[n-1][3]++;
                }
            }
        }
    }
    
   // thousandths precision
    outfile << setprecision(3) << fixed;
    A = 0;B = 0;C = 0;D = 0;
    for (int i = 0; i < outputN; i++){
        A += result[i][0];
        B += result[i][1];
        C += result[i][2];
        D += result[i][3];
        outfile << (int)result[i][0] << " " << (int)result[i][1] << " "<<(int)result[i][2] << " " <<(int)result[i][3] << " ";
        overall = (result[i][0] + result[i][3])/(result[i][0]+result[i][1]+result[i][2]+result[i][3]);
        precision = result[i][0] / (result[i][0] + result[i][1]);
        recall = result[i][0] / (result[i][0] + result[i][2]);
        f1 = (2*precision*recall)/ (precision + recall);
        if (overall != overall) overall = 0;
        if (precision != precision) precision = 0;
        if (recall != recall)recall = 0;
        if (f1 != f1)f1 = 0;
        outfile << overall << " " << precision << " "<< recall<< " "<< f1 << endl;
        
        avgOverall +=overall;
        avgPrecision +=precision;
        avgRecall += recall;
    }
    
    // micro-averaging 
    overall = (A + D)/(A + B + C + D);
    precision = A/(A+B);
    recall = A/(A+C);
    f1 = (2 * precision * recall)/(precision + recall);
    if (overall != overall) overall = 0;
    if (precision != precision) precision = 0;
    if (recall != recall)recall = 0;
    if (f1 != f1)f1 = 0;
    outfile << overall << " " << precision << " " << recall << " " << f1 << endl;
    // macro-averaging
    avgOverall /=outputN;
    avgPrecision /=outputN;
    avgRecall /=outputN;
    avgF1 = (2*avgPrecision*avgRecall)/(avgPrecision + avgRecall);
    if (avgF1 != avgF1)avgF1 = 0;
    outfile << avgOverall<< " " << avgPrecision << " " << avgRecall << " " << avgF1 << endl;
    return 0;
    }

// network-save
void neuralNetwork::save(ostream &outfile){
    outfile << setprecision(3) << fixed;
    for (int i = 0; i < this -> numLayers; i++){
        if (i != 0){
            outfile << " ";
        }
        outfile << this -> layerSize[i]-1;
    }
    outfile << endl;
    for (int i = 1; i < this -> numLayers; i ++){
        for (int j = 1; j < this -> layerSize[i]; j++){
            vector<link>::iterator it;
            for (it = this -> layers[i][j].inLinks.begin(); it != this -> layers[i][j].inLinks.end(); it ++){
                if (it != this -> layers[i][j].inLinks.begin()){
                    outfile << " ";
                }
                outfile << it -> weight;
            }
            outfile << endl;
        }
    }
}


// activation
double neuralNetwork::sigmoid(double inputValue){
    return 1.000 / (1.000 + exp(-inputValue));
}

double neuralNetwork::sigmoidPrime(double inputValue){
    return this -> sigmoid(inputValue) * (1.000 - this -> sigmoid(inputValue));
}

