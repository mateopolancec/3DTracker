#pragma clang diagnostic push
#pragma ide diagnostic ignored "performance-inefficient-string-concatenation"

/* INCLUDES FOR THIS PROJECT */
#include "ReceiveData.h"

using std::end;
using std::begin;


/* MAIN PROGRAM */
int main(int argc, const char *argv[]) {

    ReceiveData receive_data;
    receive_data.Update();

    return 0;
}

#pragma clang diagnostic pop