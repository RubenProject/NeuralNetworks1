//Gemaakt door Tim Poot (S1514113) & Ruben van der Waal (S1559451)
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

using namespace std;

const int MAX = 20;
const double ALPHA = 0.1;
const double BETA = 1.0;


// g-functie (sigmoid)
double g (double x) {
  return 1 / ( 1 + exp ( - BETA * x ) );
}//g

// afgeleide van g
double gprime (double x) {
  return BETA * g (x) * ( 1 - g (x) );
}//gprime

int main (int argc, char* argv[ ]) {

  string type;
  int inputs, hiddens;            // aantal invoer- en verborgen knopen
  double input[MAX];              // de invoer is input[1]...input[inputs]
  double inputtohidden[MAX][MAX]; // gewichten van invoerknopen 0..inputs
                                  // naar verborgen knopen 1..hiddens
  double hiddentooutput[MAX];     // gewichten van verborgen knopen 0..hiddens
                                  // naar de ene uitvoerknoop
  double inhidden[MAX];           // invoer voor de verborgen knopen 1..hiddens
  double acthidden[MAX];          // en de uitvoer daarvan
  double inoutput;                // invoer voor de ene uitvoerknoop
  double netoutput;               // en de uitvoer daarvan: de net-uitvoer
  double target;                  // gewenste uitvoer
  double error;                   // verschil tussen gewenste en 
                                  // geproduceerde uitvoer
  double delta;                   // de delta voor de uitvoerknoop
  double deltahidden[MAX];        // de delta's voor de verborgen 
                                  // knopen 1..hiddens
  int epochs;                     // aantal trainingsvoorbeelden
  int i, j, k;                    // tellertjes
  int x, y, res;
  //int seed = 1234;                // eventueel voor random-generator

  if ( argc != 5 ) {
    cout << "Gebruik: " << argv[0] << " <inputs> <hiddens> <epochs> <type>" << endl;
    return 1;
  }//if
  inputs = atoi (argv[1]);
  hiddens = atoi (argv[2]);
  epochs = atoi (argv[3]);
  type = argv[4];
  input[0] = -1;                  // invoer bias-knoop: altijd -1
  acthidden[0] = -1;              // verborgen bias-knoop: altijd -1
  srand (time(NULL));

  //TODO-1 initialiseer de gewichten random tussen -1 en 1: 
  // inputtohidden en hiddentooutput
  // rand ( ) levert geheel randomgetal tussen 0 en RAND_MAX; denk aan casten
  // let steeds op de rand-indices 0 en 1




  for ( i = 0; i <= inputs; i++ )  {
    for ( j = 1; j <= hiddens; j++ )  {
      inputtohidden[i][j] = ((double)rand()/(double)RAND_MAX)*2-1;
    }
  }

  for ( i = 0; i <= hiddens; i++ )  {
    hiddentooutput[i] = ((double)rand()/(double)RAND_MAX)*2-1;
  }

  for ( k = 0; k < epochs; k++ ) {
    inoutput = 0;

    for (i = 1; i <= hiddens; i++ ) {
      inhidden[i] = 0;
    }
    //TODO-2 lees een voorbeeld in naar input en target, of genereer dat ter plekke:
    // als voorbeeld: de XOR-functie, waarvoor geldt dat inputs = 2
if (type == "XOR") {
      x = rand ( ) % 2;
      y = rand ( ) % 2;
      res = ( x + y ) % 2;
      input[1] = (double) x;
      input[2] = (double) y;
      target = (double) res;
    } else if (type == "AND") {
      x = rand ( ) % 2;
      y = rand ( ) % 2;
      input[1] = (double) x;
      input[2] = (double) y;
      res = x & y;
      target = (double) res;
    } else if (type == "OR") {
      x = rand ( ) % 2;
      y = rand ( ) % 2;
      input[1] = (double) x;
      input[2] = (double) y;
      res = x | y;
      target = (double) res;
    } else {
      cout << "Please use \"AND\" \"OR\" or \"XOR\"." << endl;
      exit(0);
    }

    //TODO-3 stuur het voorbeeld door het netwerk
    // reken inhidden's uit, acthidden's, inoutput en netoutput
    for ( i = 0; i <= inputs; i++ )  {
      for ( j = 1; j <= hiddens; j++ )  {
        inhidden[j] += input[i] * inputtohidden[i][j];
      }
    }
    for ( i = 1; i <= hiddens; i++ )  {
      acthidden[i] = g(inhidden[i]);
    }
    for ( i = 0; i <= hiddens; i++ )  {
      inoutput += acthidden[i] * hiddentooutput[i];
    }

    netoutput = g(inoutput);

    //TODO-4 bereken error, delta, en deltahidden

    error = target - netoutput;
    delta = error * gprime(inoutput);
    for ( i = 0; i <= hiddens; i++ )  {
      deltahidden[i] = gprime(inhidden[i]) * hiddentooutput[i] * delta;
    }
    //TODO-5 update gewichten hiddentooutput en inputtohidden

      for ( i = 0; i <= hiddens; i++ )  {
        hiddentooutput[i] = hiddentooutput[i] + ALPHA * acthidden[i] * delta;
      }

      for ( i = 0; i <= inputs; i++ ) {
        for ( j = 1; j <= hiddens; j++ )  {
          inputtohidden[i][j] = inputtohidden[i][j] + ALPHA * input[i] * deltahidden[j];
        }
      }
    }//for

//-------------------------------------------------------------------------------------------------------------------
    //TODO-5 beoordeel het netwerk en rapporteer
    sqerror = 0;
    for ( k = 0; k < 100; k++ )  {
      inoutput = 0;

      for (i = 1; i <= hiddens; i++ ) {
        inhidden[i] = 0;
      }

      //TODO-2 lees een voorbeeld in naar input en target, of genereer dat ter plekke:
      // als voorbeeld: de XOR-functie, waarvoor geldt dat inputs = 2
      if (type == "XOR") {
        x = rand ( ) % 2;
        y = rand ( ) % 2;
        res = ( x + y ) % 2;
        input[1] = (double) x;
        input[2] = (double) y;
        target = (double) res;
      } else if (type == "AND") {
        x = rand ( ) % 2;
        y = rand ( ) % 2;
        input[1] = (double) x;
        input[2] = (double) y;
        res = x & y;
        target = (double) res;
      } else if (type == "OR") {
        x = rand ( ) % 2;
        y = rand ( ) % 2;
        input[1] = (double) x;
        input[2] = (double) y;
        res = x | y;
        target = (double) res;
      } else {
        cout << "Please use \"AND\" \"OR\" or \"XOR\"." << endl;
        exit(0);
      }

      //TODO-3 stuur het voorbeeld door het netwerk
      // reken inhidden's uit, acthidden's, inoutput en netoutput
      for ( i = 0; i <= inputs; i++ )  {
        for ( j = 1; j <= hiddens; j++ )  {
          inhidden[j] += input[i] * inputtohidden[i][j];
        }
      }
      for ( i = 1; i <= hiddens; i++ )  {
        acthidden[i] = g(inhidden[i]);
      }
      for ( i = 0; i <= hiddens; i++ )  {
        inoutput += acthidden[i] * hiddentooutput[i];
      }

      netoutput = g(inoutput);

      error = target - netoutput;
      sqerror += error * error;
    }
    cout << "----------------------" << endl;
    cout << "epochs: " << h * EPOCH_STEP << endl;
    cout << "square error: " << sqerror << endl;
  }
  return 0;
}//main


