/*
  CEC14 Test Function Suite for Single Objective Optimization
  Jane Jing Liang (email: liangjing@zzu.edu.cn; liangjing@pmail.ntu.edu.cn)
  Dec. 12th 2013
*/

#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <vector>
#include "cec2014.hpp"


int main() {
    int i,j,n,m,func_num;
    FILE *fpt;
    char FileName[30];
    m=2;
    n=10;
    char x_str[40];

    std::vector<vector_double> x(m, vector_double(n, 0));
    vector_double f(m, 0.0);

    for (i = 0; i < 30; i++) {
        func_num=i+1;
        CEC2014 prob = CEC2014(func_num, n);

        sprintf(FileName, "input_data/shift_data_%d.txt", func_num);
        fpt = fopen(FileName,"r");

        if (fpt==NULL) {
            printf("\n Error: Cannot open input file for reading \n");
        }

        for(j=0; j < n; j++) {
            fscanf(fpt, "%s", x_str);
            x[0][j] = atof(x_str);
            //printf("%f\n",x[0][j]);
        }

        fclose(fpt);

        for (j = 0; j < n; j++) {
            x[1][j] = 0.0;
            //printf("%f\n",x[1][j]);
        }

        for (j = 0; j < m; j++) {
            f[j] = prob.fitness(x[j]);
        }
        for (j = 0; j < m; j++) {
            printf("f%d(x[%d]) = %f,",func_num,j+1,f[j]);
        }
        printf("\n");

    }

    return 0;
}
