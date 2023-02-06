#include "neuralNetwork.h"
#include "errno.h"
#include <cstring>

using namespace std;


int main() {
    string trainfilename, testfilename, initfilename, trainedfilename, outfilename;
    ifstream trainfile, testfile, initfile, trainedfile;
    ofstream outfile;

    double learningRate;
    int epoch;
    
    string s;
    
    cout << "Train or Test?" << endl << ">";
    cin >> s;

    if (s == "train" || s == "Train"){
        cout << "Input Initial NN filename [.init]" << endl << ">";
        cin >> initfilename;
        cout << "Input training set filename [.train]" << endl << ">";
        cin >> trainfilename;
        cout << "Input output filename [.lr.epoch.trained]"<< endl << ">";
        cin >> outfilename;
        cout << "Input desired learning rate" << endl << ">";
        cin >> learningRate;
        cout << "Input desired number of epochs" << endl << ">";
        cin >> epoch;

        initfile.open(initfilename.c_str());
        trainfile.open(trainfilename.c_str());
        outfile.open(outfilename.c_str());

        if (initfile.is_open() && trainfile.is_open() && outfile.is_open()){
            neuralNetwork *test = new neuralNetwork(initfile);
            test -> train(trainfile, learningRate, epoch);
            test -> save(outfile);
        }
        else {
            fprintf(stderr, "Unable to open input (or output) files: %s", strerror(errno));
            return -1;
        }
    }

    else if (s == "test" || s == "Test"){

        cout << "Input trained file [.lr.epoch.trained]" << endl << ">";
        cin >> trainedfilename;
        cout << "Input testing set filename [.test]" << endl << ">";
        cin >> testfilename;
        cout << "Input output filename [.lr.epoch.results]" << endl << ">";
        cin >> outfilename;

        trainedfile.open(trainedfilename.c_str());
        testfile.open(testfilename.c_str());
        outfile.open(outfilename.c_str());

        if (trainedfile.is_open() && testfile.is_open() && outfile.is_open()){
            neuralNetwork *test = new neuralNetwork(trainedfile);
            test -> test(testfile, outfile);
        }
        else {
            fprintf(stderr, "Unable to open input (or output) files: %s", strerror(errno));
            return -1;
        }
    }
    else {
        cout << "Invalid input" << endl;
        return -1;
    }
    return 0;
}
