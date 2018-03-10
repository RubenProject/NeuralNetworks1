
//
// C++-programma voor neuraal netwerk (NN) met \'e\'en output-knoop
// Zie www.liacs.leidenuniv.nl/~kosterswa/AI/nnhelp.pdf
// 13 april 2016
// Compileren: g++ -Wall -O2 -o nn nn.cc
// Gebruik:    ./nn <inputs> <hiddens> <epochs>
// Voorbeeld:  ./nn 2 3 100000
//

#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <sstream>

using namespace std;


#define INPUT_N 256
#define OUTPUT_N 10
#define ALPHA 0.1
#define BETA 1.0
#define EPOCH_STEP 200

#define VEC1_D vector<double>
#define VEC2_D vector<vector<double> >


#define TRAIN_IN "../data/train_in.csv"
#define TRAIN_OUT "../data/train_out.csv"
#define TEST_IN "../data/test_in.csv"
#define TEST_OUT "../data/test_out.csv"


struct Network{
    double input[INPUT_N + 1];            
    double input_output[INPUT_N + 1][OUTPUT_N];
    double output_in[OUTPUT_N];    
    double output_out[OUTPUT_N];    
};


// g-functie (sigmoid)
double g (double x) {
  return 1 / ( 1 + exp ( - BETA * x ) );
}//g

// afgeleide van g
double gprime (double x) {
  return BETA * g (x) * ( 1 - g (x) );
}//gprime


bool load_set(string file_in, string file_out, VEC2_D& in, VEC1_D& out){
    ifstream f_in, f_out;
    f_in.open(file_in);
    string line, word;
    int i = 0;
    while (getline(f_in, line, '\n')){
        stringstream sline(line); 
        in.push_back(VEC1_D ());
        while (getline(sline, word, ',')){
            in[i].push_back(atof(word.c_str()));
        }
        i++;
    }
    f_in.close();

    f_out.open(file_out);
    while (getline(f_out, word, '\n')){
        out.push_back(atof(word.c_str()));
    }
    f_out.close();
    return in.size() == out.size();
}



bool train(Network *net, int epochs){
    VEC2_D train_in;
    VEC1_D train_out;
    double error[OUTPUT_N], delta[OUTPUT_N], target;
    int h, i, j;

    if (!load_set(TRAIN_IN, TRAIN_OUT, train_in, train_out))
        return false;

    //randomly initialize weights
    for (i = 0; i <= INPUT_N; i++){
        for (j = 0; j < OUTPUT_N; j++){
            net->input_output[i][j] = ((double)rand() / (double)RAND_MAX) * 2 - 1;
        }
    }

    net->input[0] = -1;

    //do this for a specified amount of epochs
    //feed the data set
    for (h = 0; h < (int)train_in.size(); h++){

        //set data to inputs
        for (i = 1; i <= INPUT_N; i++)
            net->input[i] = train_in[h][i-1];
        target = train_out[h];

        //evaluate
        for (i = 0; i < OUTPUT_N; i++){
            net->output_in[i] = 0;
            for (j = 0; j <= INPUT_N; j++ )  {
                net->output_in[i] += net->input_output[i][j] * net->input[j];
            }
            net->output_out[i] = g(net->output_in[i]);
        }
    

        //calculate errors 
        for (i = 0; i < OUTPUT_N; i++){
            if ((int)target == i)
                error[i] = 1 - net->output_out[i];
            else
                error[i] = net->output_out[i];
            delta[i] = error[i] * gprime(net->output_in[i]);
        }

        //update weights
        for (i = 0; i <= OUTPUT_N; i++ ){
            for (j = 0; j <= INPUT_N; j++){
                net->input_output[i][j] = net->input_output[i][j] + ALPHA * net->input[j] * delta[i];
            }
        }
    }//for
}


double test(Network *net){
    VEC2_D test_in;
    VEC1_D test_out;
    if (!load_set(TEST_IN, TEST_OUT, test_in, test_out))
        return -1.0;

    int h, i, j;
    double error[OUTPUT_N], target, sqerror = 0;

    net->input[0] = -1;

    for (h = 0; h < (int)test_in.size(); h++){

        //set data to inputs
        for (i = 1; i <= INPUT_N; i++){
            net->input[i] = test_in[h][i-1];
        }
        target = test_out[h];

        //set input buffers to zero 
        for (i = 0; i < OUTPUT_N; i++){
            net->output_in[i] = 0;
        }

        //TODO-3 stuur het voorbeeld door het netwerk
        for ( i = 0; i <= INPUT_N; i++ )  {
            for ( j = 0; j < OUTPUT_N; j++ )  {
                net->output_in[j] += net->input[i] * net->input_output[i][j];
            }
        }

        for ( i = 0; i < OUTPUT_N; i++ )  {
            net->output_out[i] = g(net->output_in[i]);
        }
    
        //calculate errors 
        for (i = 0; i < OUTPUT_N; i++){
            if ((int)target == i)
                error[i] = 1 - net->output_out[i];
            else
                error[i] = net->output_out[i];
        }

        sqerror = 0;
        for (i = 0; i < OUTPUT_N; i++){
            sqerror += error[i] * error[i];
        }
        cout << sqerror << endl;
    }
    //return square error
    return sqerror;
    
}



int main (int argc, char* argv[ ]) {
    Network *net;
    int epochs;

    if ( argc != 2 ) {
      return 1;  
    }//if

    epochs = atoi (argv[1]);

    net = new Network();

    //train network
    train(net, 1000);
    //test network
    cout << test(net) << endl;

    return 0;
}//main

