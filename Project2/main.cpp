#include "problems.h"


int main(int argc, char** argv) {
    int problem = (int)(argv[1][0]) - (int)('0'); // 'd' --> d for any digit d.
    
    switch (problem) {
        case 2:
            problem2();
            break;
        case 3:
            problem3();
            break;
        case 4:
            problem4();
            break;
        default:
            break;
    };

    return 0;
}
