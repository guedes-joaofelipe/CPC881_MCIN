"""

Copyright (c) 2017, Yarpiz (www.yarpiz.com)
All rights reserved. Please read the "license.txt" for usage terms.
__________________________________________________________________________

Project Code: YPEA127
Project Title: Implementation of Particle Swarm Optimization in Python
Publisher: Yarpiz (www.yarpiz.com)

Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)

Contact Info: sm.kalami@gmail.com, info@yarpiz.com

"""

import numpy  as np
import pandas as pd
# from copy    import copy

# Particle Swarm Optimization
def PSO(problem, MaxIter = 100, popSize = 100, c1 = 1.4962, c2 = 1.4962, w = 0.7298, wdamp = 1.0):

    # Empty Particle Template
    empty_particle = {
        'position': None,
        'velocity': None,
        'cost': None,
        'best_position': None,
        'best_cost': None,
    }

    # Extract Problem Info
    CostFunction = problem['CostFunction']
    xMin = problem['xMin']
    xMax = problem['xMax']
    nVar = problem['nVar']

    # Initialize Global Best
    gbest = {'position': None, 'cost': np.inf}

    # Create Initial Population
    popList = []
    pop = []
    for i in range(0, popSize):
        pop.append(empty_particle.copy())
        pop[i]['position'] = np.random.uniform(xMin, xMax, nVar)
        pop[i]['velocity'] = np.zeros(nVar)
        pop[i]['cost'] = CostFunction(pop[i]['position'])
        pop[i]['best_position'] = pop[i]['position'].copy()
        pop[i]['best_cost'] = pop[i]['cost']

        if pop[i]['best_cost'] < gbest['cost']:
            gbest['position'] = pop[i]['best_position'].copy()
            gbest['cost'] = pop[i]['best_cost']

    popList.append(pop)
    # PSO Loop
    for it in range(0, MaxIter):
        for i in range(0, popSize):

            pop[i]['velocity'] = w*pop[i]['velocity'] \
                + c1*np.random.rand(nVar)*(pop[i]['best_position'] - pop[i]['position']) \
                + c2*np.random.rand(nVar)*(gbest['position'] - pop[i]['position'])

            pop[i]['position'] += pop[i]['velocity']
            pop[i]['position'] = np.maximum(pop[i]['position'], xMin)
            pop[i]['position'] = np.minimum(pop[i]['position'], xMax)

            pop[i]['cost'] = CostFunction(pop[i]['position'])

            if pop[i]['cost'] < pop[i]['best_cost']:
                pop[i]['best_position'] = pop[i]['position'].copy()
                pop[i]['best_cost'] = pop[i]['cost']

                if pop[i]['best_cost'] < gbest['cost']:
                    gbest['position'] = pop[i]['best_position'].copy()
                    gbest['cost'] = pop[i]['best_cost']

        w *= wdamp
        popList.append(pop)
        # print('Iteration {}: Best Cost = {}'.format(it, gbest['cost']))

    return gbest, popList

# Generalized Opposition-based Particle Swarm Optimization
def GOPSO(problem, MaxIter = 100, popSize = 100, c1 = 1.4962, c2 = 1.4962, w = 0.7298, wdamp = 1.0):
    # Empty Particle Template
    empty_particle = {
        'position': None,
        'velocity': None,
        'cost': None,
        'best_position': None,
        'best_cost': None,
    }

    # Extract Problem Info
    CostFunction = problem['CostFunction']
    xMin = problem['xMin']
    xMax = problem['xMax']
    nVar = problem['nVar']

    # Initialize Global Best
    gbest = {'position': None, 'cost': np.inf}

    # Create Initial Population
    popList     = []
    columnNames = []
    for i in range(nVar):
        columnNames.append('position{}'.format(i))
    columnNames.extend(['velocity', 'cost', 'best_position', 'best_cost'])

    pop = pd.DataFrame(columns=columnNames)

    pop.iloc[:, :nVar] = (xMax - xMin)*np.random.random(size=(popSize, nVar)) + xMin
    pop.loc[:, 'velocity'] = np.zeros((popSize, nVar))
    pop.loc[:, 'cost']     = CostFunction(pop.iloc[:, nVar])
    print(pop)

    pop.loc[popSize:2*popSize, 'position'] = opposite_number(pop.loc[:popSize], xMin, xMax, k='random')
    pop.loc[popSize:2*popSize, 'cost']     = CostFunction(pop.loc[popSize:2*popSize, 'position'])

    pop = pop.sort_values("cost", ascending=True, inplace=False).iloc[:popSize, :]
    print(pop)
    input()

    # for i in range(0, 2*popSize, 2):
    #     pop.loc[i, 'position']      = np.random.uniform(xMin, xMax, nVar)
    #     pop.loc[i, 'velocity']      = np.zeros(nVar)
    #     pop.loc[i, 'cost']          = CostFunction(pop.loc[i, 'position'])
    #     pop.loc[i, 'best_position'] = pop.loc[i, 'position'].copy()
    #     pop.loc[i, 'best_cost']     = pop.loc[i, 'cost']

        # if pop[i]['best_cost'] < gbest['cost']:
        #     gbest['position'] = pop[i]['best_position'].copy()
        #     gbest['cost']     = pop[i]['best_cost']

        # costList.append(pop[i]['cost'])

        # # Initialize opposite population
        # pop.append(empty_particle.copy())
        # pop[i+1]['position']      = opposite_number(pop[i+1]['position'], xMin, xMax, k='random'))
        # pop[i+1]['velocity']      = np.zeros(nVar)
        # pop[i+1]['cost']          = CostFunction(pop[i+1]['position'])
        # pop[i+1]['best_position'] = pop[i+1]['position'].copy()
        # pop[i+1]['best_cost']     = pop[i+1]['cost']


    popList.append(pop)
    # PSO Loop
    for it in range(0, MaxIter):
        for i in range(0, popSize):

            pop[i]['velocity'] = w*pop[i]['velocity'] \
                + c1*np.random.rand(nVar)*(pop[i]['best_position'] - pop[i]['position']) \
                + c2*np.random.rand(nVar)*(gbest['position'] - pop[i]['position'])

            pop[i]['position'] += pop[i]['velocity']
            pop[i]['position'] = np.maximum(pop[i]['position'], xMin)
            pop[i]['position'] = np.minimum(pop[i]['position'], xMax)

            pop[i]['cost'] = CostFunction(pop[i]['position'])

            if pop[i]['cost'] < pop[i]['best_cost']:
                pop[i]['best_position'] = pop[i]['position'].copy()
                pop[i]['best_cost'] = pop[i]['cost']

                if pop[i]['best_cost'] < gbest['cost']:
                    gbest['position'] = pop[i]['best_position'].copy()
                    gbest['cost'] = pop[i]['best_cost']

        w *= wdamp
        popList.append(pop)
        # print('Iteration {}: Best Cost = {}'.format(it, gbest['cost']))

    return gbest, popList
