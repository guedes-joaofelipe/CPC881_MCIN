# CPC881 Nature-inspired Computational Methods (MCIN)

Assignment for the class of 2019.

This repository contains the implementation of several evolutionary algorithms to be tested in the [CEC2014 functions benchmark](https://www.ntu.edu.sg/home/EPNSugan/index_files/CEC2014/CEC2014.htm). The implemented and tested algorithms are: 

- [Differential Evolution (DE)](https://www.researchgate.net/publication/227242104_Differential_Evolution_-_A_Simple_and_Efficient_Heuristic_for_Global_Optimization_over_Continuous_Spaces)
- [Opposition-based Differential Evolution (ODE)](https://ieeexplore.ieee.org/document/4358759)
- [Quasi-oppositional Differential Evolution (QODE)](https://ieeexplore.ieee.org/document/4424748)

To be tested in the following functions: 

| Function | Global Minimum | Name | 
| -------- | -------------- | ---- | 
| F1       |    100         | _Rotated High Conditioned Elliptic Function_ | 
| F2       |    200         | _Rotated Bent Cigar Function_ | 
| F6       |    600         | _Shifted and Rotated Weierstrass Elliptic Function_ | 
| F7       |    700         | _Shifted and Rotated Griewank's Function_ | 
| F9       |    900         | _Shifted and Rotated Rastrigin's Function_ | 
| F14      |   1400         | _Shifted and Rotated HGBat Function_ | 


In order to consider a succesful optimization, the absolute error  between the target and obtained values to be considered is $\epsilon = |y-\hat{y}| = 10^{-8}$. The optimization process is haulted under 2 stopping criterias: 

- target-error achieved ($\epsilon < 10^{-8}$)
- Number of fitness evaluations greater than a maximum ($FEvals > MaxFEvals$)

For a D-dimensioned problem, MaxFEvals is set to be 10000*D. 

## Running the experiments

All experiments can be run in the `notebooks` folder, where a file for each algorithm was developed. In order to optimize a combination of experiments, the `experiments_parameters.xlsx` excel sheet stores parameters inputs to be consumed by the `papermill_de.py` script. 

The `papermill_de.py` script takes a combination of parameters and runs parallel processes for each line of the excel sheet. The results of the experiments are stored on jupyter notebooks at `notebooks/output_notebooks/`. This script runs with two specific python libraries: 

- [Multiprocessing](https://docs.python.org/2/library/multiprocessing.html)
- [Papermill ](https://papermill.readthedocs.io/en/latest/)

So make sure to have both of these libraries installed. 

All things considered, the following steps are needed to run the experiments:

1) Set the parameters at `experiments_parameters.xlsx` 
2) Change the `algorithm` variable at `papermill_de.py` according to the sheet with the variables in **1.** (this needs automatizing)
3) Run `python papermill_de.py`
4) Check the results in `notebooks/Output_Notebooks/`


## Results Example

For the following results, an ensemble of 51 realizations was performed and the statistics under such realizations were obtained. 

In a D=10 dimension problem, we get the following results:

![MaxFEvals for 10 dimensions](./results/General/[ALLF_10_51_30_1.0_1_False_5]mean_maxFES.jpeg "MaxFEvals for 10 dimensions")

The x-axis is a percentage of the MaxFEvals and the y-axis is the mean performance of each algorithm on the 51 experiments. Statistic for each algorithm can also be found on the following table:

| Algorithm       | Fi \- D10 | Best       | Worst      | Median     | Mean       | Std Dev    | Success Rate (%) | Elapsed Time (min) |
|-----------------|-----------|------------|------------|------------|------------|------------|--------------------|----------------------|
| DE/best/1/bin   | F2        | 4\.16E\-09 | 9\.98E\-09 | 8\.38E\-09 | 8\.12E\-09 | 1\.37E\-09 | 100\.00            | 43\.17               |
| DE/best/1/bin   | F6        | 4\.86E\-09 | 3\.46E\+00 | 1\.41E\-01 | 5\.60E\-01 | 7\.83E\-01 | 11\.76             | 324\.62              |
| DE/best/1/bin   | F1        | 7\.24E\-09 | 9\.98E\-09 | 9\.39E\-09 | 9\.17E\-09 | 6\.83E\-10 | 100\.00            | 94\.98               |
| DE/best/1/bin   | F7        | 3\.94E\-02 | 9\.80E\-01 | 1\.62E\-01 | 2\.28E\-01 | 2\.08E\-01 | 0\.00              | 491\.37              |
| DE/best/1/bin   | F14       | 7\.00E\-02 | 4\.58E\-01 | 2\.71E\-01 | 2\.76E\-01 | 7\.58E\-02 | 0\.00              | 490\.63              |
| DE/best/1/bin   | F9        | 4\.97E\+00 | 3\.80E\+01 | 1\.69E\+01 | 1\.78E\+01 | 8\.70E\+00 | 0\.00              | 490\.68              |
| ODE/best/1/bin  | F6        | 2\.13E\-09 | 4\.58E\+00 | 3\.29E\-02 | 6\.94E\-01 | 1\.01E\+00 | 29\.41             | 128\.98              |
| ODE/best/1/bin  | F1        | 4\.30E\-09 | 9\.98E\-09 | 9\.26E\-09 | 8\.57E\-09 | 1\.35E\-09 | 100\.00            | 57\.42               |
| ODE/best/1/bin  | F2        | 4\.72E\-09 | 9\.99E\-09 | 8\.56E\-09 | 8\.18E\-09 | 1\.38E\-09 | 100\.00            | 29\.03               |
| ODE/best/1/bin  | F7        | 2\.21E\-02 | 4\.56E\-01 | 1\.23E\-01 | 1\.42E\-01 | 9\.67E\-02 | 0\.00              | 154\.13              |
| ODE/best/1/bin  | F14       | 9\.11E\-02 | 4\.81E\-01 | 2\.66E\-01 | 2\.70E\-01 | 8\.88E\-02 | 0\.00              | 140\.53              |
| ODE/best/1/bin  | F9        | 6\.37E\+00 | 5\.45E\+01 | 1\.90E\+01 | 2\.06E\+01 | 1\.07E\+01 | 0\.00              | 143\.03              |
| QODE/best/1/bin | F2        | 3\.41E\-09 | 9\.99E\-09 | 8\.89E\-09 | 8\.49E\-09 | 1\.32E\-09 | 100\.00            | 56\.28               |
| QODE/best/1/bin | F1        | 5\.40E\-09 | 9\.99E\-09 | 9\.36E\-09 | 9\.06E\-09 | 1\.05E\-09 | 100\.00            | 70\.3                |
| QODE/best/1/bin | F6        | 4\.00E\-03 | 3\.62E\+00 | 1\.06E\+00 | 1\.16E\+00 | 9\.19E\-01 | 0\.00              | 106\.25              |
| QODE/best/1/bin | F7        | 9\.86E\-03 | 1\.61E\+00 | 1\.23E\-01 | 2\.37E\-01 | 3\.14E\-01 | 0\.00              | 150\.28              |
| QODE/best/1/bin | F14       | 7\.72E\-02 | 4\.99E\-01 | 3\.15E\-01 | 3\.13E\-01 | 1\.04E\-01 | 0\.00              | 148\.67              |
| QODE/best/1/bin | F9        | 3\.98E\+00 | 4\.44E\+01 | 1\.19E\+01 | 1\.39E\+01 | 7\.67E\+00 | 0\.00              | 148\.92              |
