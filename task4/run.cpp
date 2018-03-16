// compile with
// g++ run.cpp -o prog
// run with: ./prog for experiments
// or ./prog <epochs> to train for x epochs

#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <climits>

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

//prototypes
void experiment(Network *net);
bool load_set(string file_in, string file_out, VEC2_D& in, VEC1_D& out);
double test(Network *net, bool test);
bool train(Network *net, int epochs);

//sigmoid function
double g (double x) {
  return 1 / ( 1 + exp ( - BETA * x ) );
}

//derivative of sigmoid
double gprime (double x) {
  return BETA * g (x) * ( 1 - g (x) );
}


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
    double error[OUTPUT_N], delta[OUTPUT_N];
    int h, i, j, k, target;

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
    for (k = 0; k < epochs; k++){
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
                    net->output_in[i] += net->input_output[j][i] * net->input[j];
                }
                net->output_out[i] = g(net->output_in[i]);
            }
        
            //calculate errors 
            for (i = 0; i < OUTPUT_N; i++){
                if (i == target)
                    error[i] = 1 - net->output_out[i];
                else
                    error[i] = 0 - net->output_out[i];
                delta[i] = error[i] * gprime(net->output_in[i]);
            }

            //update weights
            for (i = 0; i < OUTPUT_N; i++ ){
                for (j = 0; j <= INPUT_N; j++){
                    net->input_output[j][i] = net->input_output[j][i] + ALPHA * net->input[j] * delta[i];
                }
            }
        }
    }
}


double test(Network *net, bool test){
    VEC2_D data_in;
    VEC1_D data_out;

    if (test && !load_set(TRAIN_IN, TRAIN_OUT, data_in, data_out))
        return -1.0;

    if (!test && !load_set(TEST_IN, TEST_OUT, data_in, data_out))
        return -1.0;

    int h, i, j, res, target, count = 0;
    double max;

    net->input[0] = -1;

    for (h = 0; h < (int)data_in.size(); h++){

        //set data to inputs
        for (i = 1; i <= INPUT_N; i++){
            net->input[i] = data_in[h][i-1];
        }
        target = data_out[h];

        //set input buffers to zero 
        for (i = 0; i < OUTPUT_N; i++){
            net->output_in[i] = 0;
        }

        //evaluate
        for (i = 0; i < OUTPUT_N; i++){
            net->output_in[i] = 0;
            for (j = 0; j <= INPUT_N; j++ )  {
                net->output_in[i] += net->input_output[j][i] * net->input[j];
            }
            net->output_out[i] = g(net->output_in[i]);
        }

        //check result
        max = -100;
        res = -1;
        for (i = 0; i < OUTPUT_N; i++){
            if (max <= net->output_out[i]){
                max = net->output_out[i];
                res = i;
            }
        }
        if (res == target)
            count++;

    }
    return (double)count / data_out.size();
}


void experiment(Network *net){
    ofstream f_out;
    const int AVG_RUN = 10;
    const int MAX_RUN = 10;
    double train_avg, test_avg;
    int i, j;

    f_out.open("./data.txt");
    for (i = 0; i < MAX_RUN; i++){
        cout << "run: " << i+1 << " of " << MAX_RUN << endl;
        train_avg = 0;
        test_avg = 0;
        for (j = 0; j < AVG_RUN; j++){
            net = new Network();
            train(net, i * 100); 
            train_avg += test(net, 0);
            test_avg += test(net, 1);
            delete net;
        }
        train_avg /= AVG_RUN;
        test_avg /= AVG_RUN;
        f_out << i * 100;
        f_out << ", "; 
        f_out << train_avg; 
        f_out << ", "; 
        f_out << test_avg;
        f_out << "\n";
    }
    f_out.close();
}


int main (int argc, char* argv[ ]) {
    Network *net;
    int epochs;

    if ( argc != 2 ) {
        experiment(net);
        return 0;
    }

    epochs = atoi (argv[1]);
    net = new Network();

    //train network
    cout << "training score: " << endl;
    train(net, epochs);
    cout << test(net, 0) << endl;
    //test network
    cout << "===============" << endl;
    cout << "test score: " << endl;
    cout << test(net, 1) << endl;

    return 0;
}//main

