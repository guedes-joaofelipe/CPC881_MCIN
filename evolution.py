import numpy    as np
import pandas   as pd
import pygmo    as pg
from copy       import copy

from utils      import opposite_number
from yarpiz     import PSO, GOPSO

class EvolutionaryAlgorithm:
    def __init__(self, dim=2, pop_size=10):
        self.dim            = dim
        self.pop_size       = pop_size

        self.fitnessEvals   = 0

class ParticleSwarmOptimizationSimple:
    def __init__(self, func_id, pop_size = 100, dim=10, max_iters=100):
        self.dim        = dim
        self.pop_size   = pop_size
        self.xMin       = -100        # Search space limits
        self.xMax       =  100        #
        self.vMin       = self.xMin/2 # Velocity limits
        self.vMax       = self.xMax/2 #

        self.func_id      = func_id
        self.fitnessEvals = 0

        # Debug settings
        self.c1    = 1.0
        self.c2    = 1.0
        self.w     = 1.0
        self.wdamp = 1.0

        # # Yarpiz default
        # self.c1    = 1.4962
        # self.c2    = 1.4962
        # self.w     = 0.7298
        # self.wdamp = 1.0

        self.problem = pg.problem(pg.cec2014(prob_id=self.func_id, dim=self.dim))

        self.init_states()

    def init_states(self):
        randomPositions = np.random.random((self.pop_size, self.dim))*(self.xMax - self.xMin) + self.xMin

        # Initialize population randomly
        self.population   = self.set_state(pd.DataFrame(randomPositions)).copy()
        self.velocity     = pd.DataFrame(np.zeros((self.pop_size, self.dim)))

        # Update best positions and costs found per particle
        self.previousBest = self.population.copy()

        # Update Global best position and cost
        self.update_global_best()
        return self.population

    def get_fitness(self, vector):
        '''
            Wrapper that returns fitness value for state input vector and increments
            number of fitness evaluations.

            Argument: vector. State vector of length (dim).

            Returns : Fitness for given input state as evaluated by target function.
        '''
        self.fitnessEvals +=1
        return self.problem.fitness(vector)[0]

    def update_previous_best(self, population):
        # Keep track of best position found per particle
        # If new position's fitness is greater than current best's, keep current best
        # Else, store new fitness it as the new best
        self.previousBest.where(population['Fitness'] >= self.previousBest['Fitness'], other=population)
        return self.previousBest


    def update_global_best(self):
        self.bestVector = self.previousBest.sort_values("Fitness", ascending=True, inplace=False).reset_index(drop=True).iloc[0, :]
        return self.bestVector

    def set_state(self, newPopulation, substitute='random'):
        '''
            Function to attribute new values to a population DataFrame.
            Guarantees correct fitness values for each state update.
            Includes specimen viability evaluation and treatment.

            Arguments:
                newPopulation   : Population DataFrame with new state values
                substitute      :
                    'random': re-initializes non-viable vectors;
            Returns: updatedPopulation, a population DataFrame with the input values
            checked for viability and updated fitness column.
        '''
        newPopSize = newPopulation.shape[0]
        ## Exception Treatment: Checks if states are inside search space. If not, reinitialize
        # Create comparison vector
        logicArray = np.logical_or(np.less(newPopulation, self.xMin),
                                   np.greater(newPopulation, self.xMax))

        # Substitute offending numbers with a random number from random array
        randomArray = np.random.random((newPopSize, self.dim))*(self.xMax - self.xMin) + self.xMin
        updatedPopulation = pd.DataFrame(np.where(logicArray, randomArray, newPopulation))

        # Compute new Fitness
        fitness = newPopulation.apply(self.get_fitness, axis=1).copy()
        updatedPopulation = updatedPopulation.assign(Fitness=fitness)

        return updatedPopulation

    def generation(self):
        randomNum1 = np.tile(np.random.random(size=(self.pop_size, 1)), (1, self.dim))
        randomNum2 = np.tile(np.random.random(size=(self.pop_size, 1)), (1, self.dim))

        bestPos         = self.bestVector.iloc[:-1].copy()
        currentPos      = self.population.iloc[:, :-1].copy()
        previousBestPos = self.previousBest.iloc[:, :-1].copy()

        # print("1\n", randomNum1)
        # print("2\n", randomNum2)
        # print("Best vector\n", self.bestVector)
        # print("Current velocity: \n", self.velocity)
        # print("Current pop: \n", self.population)
        # print("Previous best pos: \n", self.previousBest)

        localSearch  = randomNum1*(previousBestPos - currentPos)
        globalSearch = randomNum2*(bestPos - currentPos)

        # print(localSearch)
        # print(globalSearch)
        # Update velocity
        self.velocity = self.w*self.velocity + self.c1*localSearch + self.c2*globalSearch

        # Clip velocities to limits
        self.velocity.clip(lower=self.vMin, upper=self.vMax, inplace=True)

        # Update position
        self.population.iloc[:, :-1] = self.population.iloc[:, :-1] + self.velocity
        self.population = self.set_state(self.population.iloc[:, :-1])

        # print("New velocity: \n", self.velocity)
        # print("New pop: \n", self.population)

        # Update best values
        self.update_previous_best(self.population)
        self.update_global_best()

        return self.population


class ParticleSwarmOptimization:
    def __init__(self, func_id, pop_size = 100, dim=10, max_iters=100):
        self.dim        = dim
        self.pop_size   = pop_size
        self.xMin       = -100       # Search space limits
        self.xMax       =  100       #
        self.maxIters   = max_iters

        self.func_id      = func_id
        self.fitnessEvals = 0

        # Original parameters
        self.c1    = 2.0
        self.c2    = 2.0
        self.w     = 1.0
        self.wdamp = 1.0

        # Yarpiz default
        # self.c1    = 1.4962
        # self.c2    = 1.4962
        # self.w     = 0.7298
        # self.wdamp = 1.0

        self.problem = pg.problem(pg.cec2014(prob_id=self.func_id, dim=self.dim))

        self.psoProblem = {
                'CostFunction': self.get_fitness,
                'nVar': self.dim,
                'xMin': self.xMin,   # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
                'xMax': self.xMax,   # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
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
        gbest, particlesGen = GOPSO(self.psoProblem, MaxIter = self.maxIters, popSize = self.pop_size,
                                    c1 = self.c1, c2 = self.c2, w = self.w, wdamp = self.wdamp)

        fitnessHist = np.empty((self.maxIters, self.pop_size))
        for i in range(self.maxIters):
            particles = particlesGen[i][:]
            for j in range(self.pop_size):
                fitnessHist[i, j] = particles[j]['cost']

        fitnessDf = pd.DataFrame(fitnessHist)

        return fitnessDf



class DifferentialEvolutionSimple:
    def __init__(self, dim=2, func_id=1, pop_size=30):
        # Initialize parameters
        self.dim        = dim
        self.pop_size   = pop_size
        self.xMin       = -100       # Search space limits
        self.xMax       =  100       #

        self.func_id      = func_id
        self.fitnessEvals = 0

        # Parameters
        self.param_F    = 0.9   # Mutation parameter F
        self.cross_rate = 0.9   # Crossover probability

        # Fitness Function definition
        self.problem = pg.problem(pg.cec2014(prob_id=self.func_id, dim=self.dim))

        # Initialize population DataFrame and compute initial Fitness
        self.init_states()

    def init_states(self):
        specimen = np.random.random((self.pop_size, self.dim))*(self.xMax - self.xMin) + self.xMin

        self.population = self.set_state(pd.DataFrame(specimen)).copy()
        return self.population


    def get_fitness(self, specimen):
        '''
            Wrapper that returns fitness value for state input specimen and increments
            number of fitness evaluations.

            Argument: specimen. State vector of length (dim).

            Returns : Fitness for given input state as evaluated by target function.
        '''
        self.fitnessEvals +=1
        return self.problem.fitness(specimen)[0]

    def set_state(self, newPopulation, substitute='random'):
        '''
            Function to attribute new values to a population DataFrame.
            Guarantees correct fitness values for each state update.
            Includes specimen viability evaluation and treatment.

            Arguments:
                newPopulation   : Population DataFrame with new state values
                substitute      :
                    'random': re-initializes non-viable vectors;
            Returns: updatedPopulation, a population DataFrame with the input values
            checked for viability and updated fitness column.
        '''

        # Exception Treatment
        # Checks if states are inside search space. If not, reinitialize
        logicArray = np.logical_or(np.less(newPopulation, self.xMin),
                                   np.greater(newPopulation, self.xMax))

        # Substitute offending numbers with a random number from random array
        randomArray = np.random.random((newPopulation.shape[0], self.dim))*(self.xMax - self.xMin) + self.xMin
        updatedPopulation = pd.DataFrame(np.where(logicArray, randomArray, newPopulation))

        # Compute new Fitness
        fitness = newPopulation.apply(self.get_fitness, axis=1).copy()
        updatedPopulation = updatedPopulation.assign(Fitness=fitness)

        return updatedPopulation

    def mutation_best_1(self, indexes, bestVector):
        '''
            DE/best/1 mutation scheme
            Every specimen produces a mutated/donor vector each generation.

            Arguments:
                index   : Target vector index.
        '''
        randomVector1 = self.population.iloc[indexes[1], :-1]
        randomVector2 = self.population.iloc[indexes[2], :-1]
        return bestVector + self.param_F*(randomVector1 - randomVector2)

    def mutate_rand_1(self, indexes):
        baseVector    = self.population.iloc[indexes[0], :-1]
        randomVector1 = self.population.iloc[indexes[1], :-1]
        randomVector2 = self.population.iloc[indexes[2], :-1]
        return baseVector + self.param_F*(randomVector1 - randomVector2)

    def mutation(self, mutation_scheme):
        def make_index_list(index):
            indexList = list(range(0, self.pop_size))
            indexList.remove(index)
            return np.random.choice(indexList, size=3, replace=False)

        randomNums = np.array(list(map(make_index_list,
                                          list(range(0, self.pop_size)))))

        if mutation_scheme == self.mutation_best_1:
            bestVector    = self.population.iloc[self.population.idxmin(axis=0)[-1], :-1]
            self.mutatedPopulation = pd.DataFrame(np.apply_along_axis(mutation_scheme, 1,
                                                                      randomNums, bestVector))
        else:
            self.mutatedPopulation = pd.DataFrame(np.apply_along_axis(mutation_scheme, 1, randomNums))
        return self.mutatedPopulation

    def generation(self):
        ## Mutation
        self.mutation(self.mutation_best_1)

        ## Crossover
        # randomArray: Roll probability for each dimension of every specimen
        # randomK:     Sample one random integer K from [0, dim] for each specimen
        # maskArray:   Mask array for logical comparisons
        randomArray = np.random.rand(self.pop_size, self.dim)
        randomK     = np.random.randint(0, self.dim, size=self.pop_size)
        maskArray   = np.arange(self.dim)

        # Reshape and tile arrays
        randomK       = np.tile(randomK[:, np.newaxis], (1, self.dim))
        maskArray     = np.tile(maskArray[np.newaxis, :], (self.pop_size, 1))

        # Substitute mutatedPopulation values if
        #   (randomArray <= cross_rate) OR (randomK == column)
        #   In other words, Keep old values only if (randomArray > cross_rate) AND (randomK != column)                                   other=self.mutatedPopulation)
        newPopulation = pd.DataFrame(np.where(np.logical_or(np.less_equal(randomArray, self.cross_rate),
                                                             np.equal(randomK, maskArray)),
                                              self.mutatedPopulation, self.population.iloc[:, :-1]))
        # Compute new fitness values and treat exceptions
        self.trialPopulation = self.set_state(newPopulation)

        # Selection: substitute Trial Vector only if it has smaller Fitness than original vector
        self.population = self.population.where(self.population["Fitness"] < self.trialPopulation["Fitness"],
                                                other=self.trialPopulation)
        return self.population

class OppositionDifferentialEvolutionSimple(DifferentialEvolutionSimple):
    def __init__(self, dim=2, func_id=1, pop_size=100):
        super().__init__(dim=dim, func_id=func_id, pop_size=pop_size)

        # Parameters
        self.param_F    = 0.5   # Mutation parameter F
        self.cross_rate = 0.9   # Crossover probability
        self.jump_rate  = 0.3

    def init_states(self):
        '''
            Randomly initialize states values following a uniform initialization
            between limits xMax and xMin. Compute opposite population with opposite
            state values. Initial population will be composed of best individuals
            of set [population, opposite_population]

            self.population is a DataFrame with one row per individual, one column per
            dimension and one extra for fitness values.

            Arguments: None

            Returns: self.population, a DataFrame with dimensions (population size)
            by (dimension + 1).
        '''
        # Initialize population uniformly over search space
        pop = np.random.random((self.pop_size, self.dim))*(self.xMax - self.xMin) + self.xMin

        # Compute opposite population and concatenate with original population
        initPopulation = pd.DataFrame(np.concatenate((pop, opposite_number(pop, self.xMin, self.xMax))))
        initPopulation  = self.set_state(initPopulation, substitute='random')

        # Keep only the fittest from set [pop, opposite_pop]
        self.population = initPopulation.sort_values("Fitness", ascending=True, inplace=False).iloc[:self.pop_size, :]
        self.population = self.population.reset_index(drop=True)
        return self.population.copy()

    def generation_jumping(self):
        '''
            Compute an opposite population to self.population and keep fittest specimens.
            Limits are given by each variables minimum and maximum values, instead
            of using the search space limits.
        '''
        # Create arrays containing min and max values per variable with same
        # shape as self.population
        infLimit = self.population.iloc[:, :-1].min(axis=0).values
        supLimit = self.population.iloc[:, :-1].max(axis=0).values

        # Reshape and tile arrays
        infLimit = np.tile(infLimit[np.newaxis, :], (self.pop_size, 1))
        supLimit = np.tile(supLimit[np.newaxis, :], (self.pop_size, 1))

        oppositePop = opposite_number(self.population.iloc[:, :-1], infLimit, supLimit)
        oppositePop = self.set_state(oppositePop, substitute='random')

        expandedPopulation = pd.concat([oppositePop, self.population], axis=0, ignore_index=True)

        # Keep only the fittest from set [pop, opposite_pop]
        self.population = expandedPopulation.sort_values("Fitness", ascending=True, inplace=False).reset_index(drop=True)
        self.population = self.population.iloc[:self.pop_size, :].reset_index(drop=True)

        return self.population

    def generation(self):
        super().generation()
        if np.random.rand() < self.jump_rate:
            # print("JUMPED")
            self.generation_jumping()

        return self.population


class DifferentialEvolution(EvolutionaryAlgorithm):
    def __init__(self, dim=2, func_id=1, pop_size=30):
        # Initialize superclass EvolutionaryAlgorithm
        super().__init__(dim=dim, pop_size=pop_size)
        self.xMin       = -100       # Search space limits
        self.xMax       =  100       #

        self.func_id    = func_id

        # Parameters
        self.param_F    = [0.8, 0.9]   # Mutation parameter F
                                # CURRENTLY MUST BE A LIST OF CHOICES FOR F
        self.cross_rate = 0.9   # Crossover probability

        # Fitness Function definition
        if self.func_id < 23:
            self.problem = pg.problem(pg.cec2014(prob_id=self.func_id, dim=self.dim))
        else:
            raise NotImplementedError("f_{:2d} not yet implemented".format(self.func_id))
            return -1

        # Initialize population DataFrame and compute initial Fitness
        self.init_states()

    def init_states(self):
        '''
            Randomly initialize states and sigma values following a uniform initialization
            between limits xMax and xMin. Assign state and fitness values to self.population.

            population is a DataFrame with one row per individual, one column per
            dimension and one extra for fitness values.

            Arguments: None

            Returns: self.population, DataFrame with dimensions (population size) by (dimension + 1).
        '''
        specimen = np.random.random((self.pop_size, self.dim))*(self.xMax - self.xMin) + self.xMin

        initPopulation = pd.DataFrame(specimen)

        self.population = self.set_state(initPopulation, substitute='random')
        return self.population.copy()

    def set_state(self, newPopulation, substitute='random'):
        '''
            Function to attribute new values to a population DataFrame.
            Guarantees correct fitness values for each state update.
            Includes specimen viability evaluation and treatment.

            Arguments:
                newPopulation   : Population DataFrame with new state values
                substitute      : Indicated what to do with non-viable specimens.
                Options are as follows:
                    'random': re-initializes non-viable vectors;
                    'none'  : substitutes non-viable vectors with None over all dimensions;
                    'edge'  : clips non-viable vectors to nearest edge of search space.

            Returns: updatedPopulation, a population DataFrame with the input values
            checked for viability and updated fitness column.
        '''
        # TODO: Split set_state and Exception Treatment in different methods
        # Exception Treatment
        # Checks if states are inside search space. If not, treat exceptions
        logicArray = np.logical_or(np.less(newPopulation, self.xMin),
                                   np.greater(newPopulation, self.xMax))

        # Case/Switch python equivalent
        # Select Exception treatment type
        choices = {
            'random':
                pd.DataFrame(np.where(logicArray, np.random.random((self.pop_size, self.dim))*(self.xMax - self.xMin) + self.xMin, newPopulation)),
            'none':
                pd.DataFrame(np.where(logicArray, np.NaN, newPopulation)),
            'edge':
                newPopulation.clip_lower(self.xMin).clip_upper(self.xMax)
        }
        updatedPopulation = choices.get(substitute, KeyError("Please select a valid substitute key")).copy()

        # Compute new Fitness
        fitness = newPopulation.apply(self.get_fitness, axis=1).copy()
        updatedPopulation = updatedPopulation.assign(Fitness=fitness)

        # Check if there is any NaN or None value in the population
        if updatedPopulation.isna().any().any():
            # Print number of NA values per specimen
            print("\nWarning: NA values found in population\n")
            print(updatedPopulation.drop(labels="Fitness", axis=1).isna().sum(axis=1))
            input()

        return updatedPopulation.copy()

    def get_fitness(self, specimen):
        '''
            Wrapper that returns fitness value for state input specimen and increments
            number of fitness evaluations.

            Argument: specimen. State vector of length (dim).

            Returns : Fitness for given input state as evaluated by target function.
        '''
        self.fitnessEvals +=1
        return self.problem.fitness(specimen)[0]

    def mutation_best_1(self, indexes):
        '''
            DE/best/1 mutation scheme
            Every specimen produces a mutated/donor vector each generation.

            Arguments:
                index   : Target vector index.
        '''
        bestVector    = self.population.iloc[self.population.idxmin(axis=0)[-1], :-1]
        randomVector1 = self.population.iloc[indexes[1], :-1]
        randomVector2 = self.population.iloc[indexes[2], :-1]
        return bestVector + self.param_F*(randomVector1 - randomVector2)

    def mutation_rand_1(self, indexes):
        '''
            DE/rand/1 mutation scheme
            Every baseSpecimen produces a mutated/donor vector each generation.

            Arguments:
                baseSpecimen, index are mantained for compatibility.
                param_F: F parameter. Must be a positive real number.
        '''
        baseVector    = self.population.iloc[indexes[0], :-1]
        randomVector1 = self.population.iloc[indexes[1], :-1]
        randomVector2 = self.population.iloc[indexes[2], :-1]
        return baseVector + self.param_F*(randomVector1 - randomVector2)

    def mutation(self, mutation_scheme):
        def make_index_list(index):
            indexList = list(range(0, self.pop_size))
            indexList.remove(index)
            return np.random.choice(indexList, size=3, replace=False)

        # # Choose F from a list defined by self.param_f
        # listParamF = np.random.choice(self.param_F, size=self.pop_size, replace=True)

        # Create list of random indexes
        randomIndexes = np.array(list(map(make_index_list, list(range(0, self.pop_size)))))

        # Mutate every specimen using scheme passed to mutationScheme
        self.mutatedPopulation = pd.DataFrame(np.apply_along_axis(mutation_scheme, 1, randomIndexes))

        return self.mutatedPopulation

    def crossover_binomial(self):
        '''
            DE binomial crossover. Compose a trial vector based on each mutated
            vector and its corresponding parent. The new vector is composed following:

                u_ij = v_ij,  if j == K or rand(0,1) <= cross_rate
                       x_ij,  else

            The trial vector will use the mutated value if probability is within
            cross_rate or its columns is randomly selected by K = randint(0, dim).

            Returns self.trialPopulation DataFrame with updated fitness column
        '''
        ## Crossover
        # randomArray: Roll probability for each dimension of every specimen
        # randomK:     Sample one random integer K from [0, dim] for each specimen
        # maskArray:   Mask array for logical comparisons
        randomArray = np.random.rand(self.pop_size, self.dim)
        randomK     = np.random.randint(0, self.dim, size=self.pop_size)
        maskArray   = np.arange(self.dim)

        # Reshape and tile arrays
        randomK       = np.tile(randomK[:, np.newaxis], (1, self.dim))
        maskArray     = np.tile(maskArray[np.newaxis, :], (self.pop_size, 1))

        # Substitute mutatedPopulation values if
        #   (randomArray <= cross_rate) OR (randomK == column)
        #   In other words, Keep old values only if (randomArray > cross_rate) AND (randomK != column)                                   other=self.mutatedPopulation)
        newPopulation = pd.DataFrame(np.where(np.logical_or(np.less_equal(randomArray, self.cross_rate),
                                                             np.equal(randomK, maskArray)),
                                              self.mutatedPopulation, self.population.iloc[:, :-1]))

        # Compute new fitness values and treat exceptions
        self.trialPopulation = self.set_state(newPopulation, substitute='random')

        return self.trialPopulation

    def survivor_selection(self):
        '''
            DE survivor selection. A mutant vector is carried over to the next
            generation if its fitness is better or equal than its parent's.

                x_i(t+1) = u_i(t), if f(u_i(t)) <= f(x_i(t))
                           x_i(t), else

            Returns updated self.population DataFrame
        '''

        # Selection: compare Fitness of Trial and Donor vectors. Substitute for new
        # values only if fitness decreases
        self.population = self.population.where(self.population["Fitness"] < self.trialPopulation["Fitness"],
                                                other=self.trialPopulation)

        return self.population

    def generation(self):
        self.mutation(self.mutation_rand_1)
        self.crossover_binomial()
        self.survivor_selection()

        return self.population

class OppositionDifferentialEvolution(DifferentialEvolution):
    def __init__(self, dim=2, func_id=1, pop_size=100):
        # Initialize superclass DifferentialEvolution
        self.xMin       = -100       # Search space limits
        self.xMax       =  100       #
        super().__init__(dim=dim, pop_size=pop_size)

        # Parameters
        self.param_F    = 0.5   # Mutation parameter F
        self.cross_rate = 0.9   # Crossover probability
        self.jump_rate  = 0.3   # Generation jumping probability

    def init_states(self):
        '''
            Randomly initialize states values following a uniform initialization
            between limits xMax and xMin. Compute opposite population with opposite
            state values. Initial population will be composed of best individuals
            of set [population, opposite_population]

            self.population is a DataFrame with one row per individual, one column per
            dimension and one extra for fitness values.

            Arguments: None

            Returns: self.population, a DataFrame with dimensions (population size)
            by (dimension + 1).
        '''
        # Initialize population uniformly over search space and compute opposite population
        pop = np.random.random((self.pop_size, self.dim))*(self.xMax - self.xMin) + self.xMin
        oppositePop = opposite_number(pop, self.xMin, self.xMax)

        initPopulation = pd.DataFrame(np.concatenate((pop, oppositePop)))

        initPopulation  = self.set_state(initPopulation, substitute='random')

        # Keep only the fittest from set [pop, opposite_pop]
        self.population = initPopulation.sort_values("Fitness", ascending=True, inplace=False).iloc[:self.pop_size, :]
        self.population = self.population.reset_index(drop=True)
        return self.population.copy()

    def generation_jumping(self):
        '''
            Compute an opposite population to self.population and keep fittest specimens.
            Limits are given by each variables minimum and maximum values, instead
            of using the search space limits.
        '''
        # Create arrays containing min and max values per variable with same
        # shape as self.population
        infLimit = self.population.iloc[:, :-1].min(axis=0).values
        supLimit = self.population.iloc[:, :-1].max(axis=0).values

        # Reshape and tile arrays
        infLimit = np.tile(infLimit[np.newaxis, :], (self.pop_size, 1))
        supLimit = np.tile(supLimit[np.newaxis, :], (self.pop_size, 1))

        oppositePop = opposite_number(self.population.iloc[:, :-1], infLimit, supLimit)
        oppositePop = self.set_state(oppositePop, substitute='random')

        expandedPopulation = pd.concat([oppositePop, self.population], axis=0, ignore_index=True)

        # Keep only the fittest from set [pop, opposite_pop]
        self.population = expandedPopulation.sort_values("Fitness", ascending=True, inplace=False).reset_index(drop=True)
        self.population = self.population.iloc[:self.pop_size, :].reset_index(drop=True)

        return self.population

    def generation(self):
        self.mutation(self.mutation_rand_1)
        self.crossover_binomial(self.mutatedPopulation)
        self.survivor_selection(self.trialPopulation)

        if np.random.random() <= self.jump_rate:
            # print("Gen JUMPED")
            self.generation_jumping()

        return self.population


class EvolutionStrategy:
    def __init__(self, dim=2, func_id=1, pop_size=20):
        self.dim        = dim        # Problem dimensionality
        self.xMin       = -100       # Search space limits
        self.xMax       =  100       #
        self.pop_size   = pop_size   # Population size
        self.func_id    = func_id
        self.fitnessEvals= 0          # Initial fitness evaluations

        # Algorithm parameters
        self.minSigma        = 0.0001
        self.tau1            = 4*1/(np.sqrt(2*self.pop_size))
        self.tau2            = 4*1/(np.sqrt(2*np.sqrt(self.pop_size)))
        self.mutateProb      = 1.0
        self.numTourneyPlays = 10

        # Fitness Function definition
        if self.func_id < 23:
            self.problem = pg.problem(pg.cec2014(prob_id=self.func_id, dim=self.dim))
        else:
            raise ValueError("f_{:2d} not yet implemented".format(self.func_id))
            return -1

        # Initialize population DataFrame and compute initial Fitness
        self.init_states()


    def init_states(self):
        '''
            Randomly initialize states and sigma values following a uniform initialization
            between limits xMax and xMin. Assign state and fitness values to self.population.

            population is a DataFrame with one row per individual and two columns per
            dimension and one extra for fitness values.

            Arguments: None

            Returns: self.population, DataFrame with dimensions (population size) by (2*dimension + 1).
        '''

        x = np.zeros((self.pop_size, 2*self.dim))

        x[:, :self.dim]   = np.random.random((self.pop_size, self.dim))*(self.xMax - self.xMin) + self.xMin
        x[:, self.dim:]   = np.random.random((self.pop_size, self.dim))

        initPopulation = pd.DataFrame(x)

        self.population = self.set_state(initPopulation)
        return self.population.copy()

    def get_fitness(self, x):
        '''
            Wrapper that returns fitness value for state input x and increments
            number of fitness evaluations.

            Argument: x. State vector of length (dim).

            Returns : Fitness for given input state as evaluated by target function.
        '''
        self.fitnessEvals +=1
        return self.problem.fitness(x)[0]

    def set_state(self, newPopulation):
        '''
            Function to attribute new values to a population DataFrame.
            Guarantees correct fitness values for each state update.

            Arguments:
                newPopulation   : Population DataFrame with new state values

            Returns: updatedPopulation, updated population DataFrame.
        '''
        # Checks if states are inside search space
        # If not, randomly initialize them
        logicArray = np.logical_or(np.less(newPopulation.iloc[:, :self.dim], self.xMin),
                                   np.greater(newPopulation.iloc[:, :self.dim], self.xMax))

        newPopulation.iloc[:, :self.dim] = np.where(logicArray,
        np.random.random()*(self.xMax - self.xMin) + self.xMin, newPopulation.iloc[:, :self.dim])

        updatedPopulation = newPopulation

        # Compute new Fitness
        fitness = newPopulation.iloc[:, :self.dim].apply(self.get_fitness, axis=1).copy()
        updatedPopulation = updatedPopulation.assign(Fitness=fitness)

        return updatedPopulation.copy()

    ## Mutate state and sigma values
    def mutate_uncorr_multistep(self, x, tau1, tau2, mutateProb=1.0):
        '''
        Apply gaussian perturbation to state values as follows:
        x_i_new     = x_i + sigma_i * gauss1_i
        sigma_i_new = sigma_i * exp(tau1 * gaussNoise) * exp(tau2 * gauss2_i)

        tau1 = 1/sqrt(2*n)
        tau2 = 1/sqrt(2*sqrt(n))

        Where
        gauss_i     are gaussian samples generated for each element
        gaussNoise  is gaussian noise used for all elements
        n           is population size
        '''
        # Evolutionary Strategy algorithms always perform mutation
        if (mutateProb > 1.0) or (mutateProb < 0.0):
            raise ValueError("Mutate probability must be a number between 0 and 1")

        newState = copy(x)
        randomNum = np.random.rand()
        if randomNum < mutateProb:
            newState = np.zeros(np.shape(x))

            # Sigma mutation
            gaussNoise = np.random.normal()
            sigmaNoise = np.random.normal(size=self.dim)

            oldSigma = x[self.dim:]
            newSigma = oldSigma*np.exp(tau1*gaussNoise)*np.exp(tau2*sigmaNoise)

            # Check for sigmas below minimum limit
            newSigma = np.where(newSigma < self.minSigma, oldSigma, newSigma)

            # State mutation
            stateNoise = np.random.normal(size=self.dim)

            newState[:self.dim] = x[:self.dim] + newSigma*stateNoise
            newState[self.dim:] = newSigma


        return newState

    def mutate(self, mutateProb=1.0):
        def mutate_row(x):
            newX = self.mutate_uncorr_multistep(x, self.tau1, self.tau2)
            return pd.Series(newX)

        newPop = self.population.drop(labels="Fitness", axis=1).apply(mutate_row, axis=1)

        self.childrenPopulation = self.set_state(newPop)
        return self.childrenPopulation.copy()

    def generation(self):
        self.mutate(mutateProb=self.mutateProb)
        self.survivor_selection_tourney(numPlays=self.numTourneyPlays)

        return self.population

    def survivor_selection_tourney(self, numPlays=10):
        '''
            Tournament selection
            Every specimen competes against each other in numPlays (q) = 10 plays.
            In each play, the specimen with greater Fitness wins. After each play, the score is updated for each specimen.
                Win:    +1
                Draw:    0
                Lose:   -2

            After the Tournament ends, the highest scoring specimens are selected until population size is filled.
        '''
        winScore  = +1
        drawScore =  0
        loseScore = -2

        # Selection with parents and children
        tourneyPop = pd.concat([self.childrenPopulation, self.population], axis=0, ignore_index=True)

        numParticipants = tourneyPop.shape[0] # Num parents + Num children

        tourneyPop["Score"] = np.zeros(numParticipants)

        for i in range(numParticipants):
            opponents = np.random.randint(0, numParticipants, size=numPlays)

            currList = np.ones(numPlays)*tourneyPop.loc[i, "Fitness"]

            oppList  = tourneyPop.loc[opponents, "Fitness"].values

            ## Score changes of current contestant
            scoreList = np.zeros(numPlays)
            scoreList += np.where(currList < oppList,  winScore, 0)
            scoreList += np.where(currList > oppList,  loseScore, 0)
            ## Uncomment for drawScore != 0
            # scoreList += np.where(currList == oppList, drawScore, 0)

            ## Score changes of opponents
            # Not implemented, probably doesn't change the outcome

            tourneyPop.loc[i, "Score"] += np.sum(scoreList)

        # Sort individuals by tourney score and keep the best
        tourneyPop.sort_values("Score", ascending=False, inplace=True)
        newPopulation = tourneyPop.iloc[:self.pop_size, :].drop(labels="Score", axis=1).copy().reset_index(drop=True)

        self.population = self.set_state(newPopulation)

        return self.population.copy()

class EvolutionStrategyMod(EvolutionaryAlgorithm):
    def __init__(self, dim=2, func_id=1, pop_size=100):
        super().__init__(dim, pop_size)

        # self.dim        = dim        # Problem dimensionality
        # self.pop_size   = pop_size   # Population size
        self.xMin       = -100       # Search space limits
        self.xMax       =  100       #
        self.min_pop    = int(pop_size*0.1)
        self.popReduce  = 0

        self.func_id    = func_id
        # self.fitnessEvals= 0          # Initial fitness evaluations

        # Algorithm parameters
        self.minSigma        = 0.0001
        self.tau1            = 4*1/(np.sqrt(2*self.pop_size))
        self.tau2            = 4*1/(np.sqrt(2*np.sqrt(self.pop_size)))
        self.mutateProb      = 1.0
        self.numTourneyPlays = 10

        # Fitness Function definition
        if self.func_id < 23:
            self.problem = pg.problem(pg.cec2014(prob_id=self.func_id, dim=self.dim))
        else:
            raise ValueError("f_{:2d} not yet implemented".format(self.func_id))
            return -1

        # Initialize population DataFrame and compute initial Fitness
        self.init_states()

    def init_states(self):
        '''
            Randomly initialize states and sigma values following a uniform initialization
            between limits xMax and xMin. Assign state and fitness values to self.population.

            population is a DataFrame with one row per individual and two columns per
            dimension and one extra for fitness values.

            Arguments: None

            Returns: self.population, DataFrame with dimensions (population size) by (2*dimension + 1).
        '''

        x = np.zeros((self.pop_size, 2*self.dim))

        x[:, :self.dim]   = np.random.random((self.pop_size, self.dim))*(self.xMax - self.xMin) + self.xMin
        x[:, self.dim:]   = np.random.random((self.pop_size, self.dim))

        initPopulation = pd.DataFrame(x)

        self.population = self.set_state(initPopulation)
        return self.population.copy()

    def get_fitness(self, x):
        '''
            Wrapper that returns fitness value for state input x and increments
            number of fitness evaluations.

            Argument: x. State vector of length (dim).

            Returns : Fitness for given input state as evaluated by target function.
        '''
        self.fitnessEvals +=1
        return self.problem.fitness(x)[0]

    def set_state(self, newPopulation):
        '''
            Function to attribute new values to a population DataFrame.
            Guarantees correct fitness values for each state update.

            Arguments:
                newPopulation   : Population DataFrame with new state values

            Returns: updatedPopulation, updated population DataFrame.
        '''
        # Checks if states are inside search space
        # If not, randomly initialize them
        logicArray = np.logical_or(np.less(newPopulation.iloc[:, :self.dim], self.xMin),
                                   np.greater(newPopulation.iloc[:, :self.dim], self.xMax))

        newPopulation.iloc[:, :self.dim] = np.where(logicArray,
        np.random.random()*(self.xMax - self.xMin) + self.xMin, newPopulation.iloc[:, :self.dim])

        updatedPopulation = newPopulation

        # Compute new Fitness
        fitness = newPopulation.iloc[:, :self.dim].apply(self.get_fitness, axis=1).copy()
        updatedPopulation = updatedPopulation.assign(Fitness=fitness)

        return updatedPopulation.copy()

    ## Mutate state and sigma values
    def mutate_uncorr_multistep(self, x, tau1, tau2, mutateProb=1.0):
        '''
        Apply gaussian perturbation to state values as follows:
        x_i_new     = x_i + sigma_i * gauss1_i
        sigma_i_new = sigma_i * exp(tau1 * gaussNoise) * exp(tau2 * gauss2_i)

        tau1 = 1/sqrt(2*n)
        tau2 = 1/sqrt(2*sqrt(n))

        Where
        gauss_i     are gaussian samples generated for each element
        gaussNoise  is gaussian noise used for all elements
        n           is population size
        '''
        # Evolutionary Strategy algorithms always perform mutation
        if (mutateProb > 1.0) or (mutateProb < 0.0):
            raise ValueError("Mutate probability must be a number between 0 and 1")

        newState = copy(x)
        randomNum = np.random.rand()
        if randomNum < mutateProb:
            newState = np.zeros(np.shape(x))

            # Sigma mutation
            gaussNoise = np.random.normal()
            sigmaNoise = np.random.normal(size=self.dim)

            oldSigma = x[self.dim:]
            newSigma = oldSigma*np.exp(tau1*gaussNoise)*np.exp(tau2*sigmaNoise)

            # Check for sigmas below minimum limit
            newSigma = np.where(newSigma < self.minSigma, oldSigma, newSigma)

            # State mutation
            stateNoise = np.random.normal(size=self.dim)

            newState[:self.dim] = x[:self.dim] + newSigma*stateNoise
            newState[self.dim:] = newSigma


        return newState

    def mutate(self, mutateProb=1.0):
        def mutate_row(x):
            newX = self.mutate_uncorr_multistep(x, self.tau1, self.tau2)
            return pd.Series(newX)

        newPop = self.population.drop(labels="Fitness", axis=1).apply(mutate_row, axis=1)

        self.childrenPopulation = self.set_state(newPop)
        return self.childrenPopulation.copy()

    def generation(self):
        self.mutate(mutateProb=self.mutateProb)
        self.survivor_selection_tourney(numPlays=self.numTourneyPlays)

        return self.population

    def survivor_selection_tourney(self, numPlays=10):
        '''
            Tournament selection
            Every specimen competes against each other in numPlays (q) = 10 plays.
            In each play, the specimen with greater Fitness wins. After each play, the score is updated for each specimen.
                Win:    +1
                Draw:    0
                Lose:   -2

            After the Tournament ends, the highest scoring specimens are selected until population size is filled.
        '''
        winScore  = +1
        drawScore =  0
        loseScore = -2

        # Selection with parents and children
        tourneyPop = pd.concat([self.childrenPopulation, self.population], axis=0, ignore_index=True)

        numParticipants = tourneyPop.shape[0] # Num parents + Num children

        tourneyPop["Score"] = np.zeros(numParticipants)

        for i in range(numParticipants):
            opponents = np.random.randint(0, numParticipants, size=numPlays)

            currList = np.ones(numPlays)*tourneyPop.loc[i, "Fitness"]

            oppList  = tourneyPop.loc[opponents, "Fitness"].values

            ## Score changes of current contestant
            scoreList = np.zeros(numPlays)
            scoreList += np.where(currList < oppList,  winScore, 0)
            scoreList += np.where(currList > oppList,  loseScore, 0)
            ## Uncomment for drawScore != 0
            # scoreList += np.where(currList == oppList, drawScore, 0)

            ## Score changes of opponents
            # Not implemented, probably doesn't change the outcome

            tourneyPop.loc[i, "Score"] += np.sum(scoreList)

        # Sort individuals by tourney score and keep the best
        tourneyPop.sort_values("Score", ascending=False, inplace=True)

        # Reduce population size every even generation
        if self.popReduce == 0:
            self.popReduce = 1
        elif (self.popReduce == 1) and (self.pop_size > self.min_pop):
            self.popReduce = 0
            self.pop_size = self.pop_size - 1

        newPopulation = tourneyPop.iloc[:self.pop_size, :].drop(labels="Score", axis=1).copy().reset_index(drop=True)

        self.population = self.set_state(newPopulation)

        return self.population.copy()


if __name__ == "__main__":
    pass
