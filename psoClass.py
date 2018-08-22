from yarpiz import PSO
import numpy    as np
import pandas   as pd
import pygmo    as pg
from copy       import copy

from utils      import opposite_number


class ParticleSwarmOptimization:
    def __init__(self, func_id, pop_size = 100, dim=10,
                c1 = 1.4962, c2 = 1.4962, w = 0.7298, wdamp = 1.0):
        self.dim        = dim
        self.pop_size   = pop_size
        self.xMin       = -100       # Search space limits
        self.xMax       =  100       #

        self.func_id      = func_id
        self.fitnessEvals = 0

        self.problem = pg.problem(pg.cec2014(prob_id=self.func_id, dim=self.dim))

        self.psoProblem = {
                'CostFunction': self.get_fitness,
                'nVar': self.dim,
                'VarMin': self.xMin,   # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
                'VarMax': self.xMax,    # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
            }

    def get_fitness(self, vector):
        '''
            Wrapper that returns fitness value for state input vector and increments
            number of fitness evaluations.

            Argument: vector. State vector of length (dim).

            Returns : Fitness for given input state as evaluated by target function.
        '''
        self.fitnessEvals +=1
        return self.problem.fitness(vector)[0]

    def all_generations(self):
        maxIter = round(self.dim*10000/self.pop_size)
        gbest, self.population = PSO(self.psoProblem, MaxIter = 100, PopSize = self.pop_size,
                                    c1 = 1.4962, c2 = 1.4962, w = 0.7298, wdamp = 1.0)

        return self.population
