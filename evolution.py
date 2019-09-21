import numpy    as np
import pandas   as pd
import pygmo    as pg

from sklearn.cluster import KMeans
from copy       import copy

from utils      import getOppositeNumber, getQuasiOppositeNumber, get_solution
from yarpiz     import PSO, GOPSO
from population import Population

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

        # # Debug
        # self.c1    = 1.0
        # self.c2    = 1.0
        # self.w     = 1.0
        # self.wdamp = 1.0

        # GOPSO default
        self.c1    = 1.49618
        self.c2    = 1.49618
        self.w     = 0.72984
        self.wdamp = 1.0

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
        self.previousBest = self.previousBest.where(population['Fitness'] >= self.previousBest['Fitness'], other=population)
        return self.previousBest


    def update_global_best(self):
        minIndex = self.previousBest["Fitness"].idxmin()

        # Record best vector and expanded dimension
        self.bestVector = self.previousBest.iloc[minIndex, :-1].values
        self.bestVector = pd.DataFrame(np.expand_dims(self.bestVector, axis=0))

        # Assign fitness column
        self.bestVector["Fitness"] = self.previousBest.iloc[minIndex, -1]
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
        if (newPopSize > 1):
            fitness = newPopulation.apply(self.get_fitness, axis=1).copy()
            updatedPopulation = updatedPopulation.assign(Fitness=fitness)
        else:
            updatedPopulation = updatedPopulation.assign(Fitness=self.get_fitness(updatedPopulation))

        return updatedPopulation

    def generation(self):
        randomNum1 = np.tile(np.random.random(size=(self.pop_size, 1)), (1, self.dim))
        randomNum2 = np.tile(np.random.random(size=(self.pop_size, 1)), (1, self.dim))

        bestPos         = self.bestVector.iloc[0, :-1].copy()
        currentPos      = self.population.iloc[:, :-1].copy()
        previousBestPos = self.previousBest.iloc[:, :-1].copy()

        localSearch  = randomNum1*(previousBestPos - currentPos)
        globalSearch = randomNum2*(bestPos - currentPos)

        # Update velocity
        self.velocity = self.w*self.velocity + self.c1*localSearch + self.c2*globalSearch

        # Clip velocities to limits
        self.velocity.clip(lower=self.vMin, upper=self.vMax, inplace=True)

        # Update position
        self.population.iloc[:, :-1] = self.population.iloc[:, :-1] + self.velocity
        self.population = self.set_state(self.population.iloc[:, :-1])

        # Update best values
        self.update_previous_best(self.population)
        self.update_global_best()

        return self.population

class GOParticleSwarmOptimizationSimple(ParticleSwarmOptimizationSimple):
    def __init__(self, func_id, pop_size = 100, dim=10, max_iters=100):
        # Call ancestor initialization
        super().__init__(func_id, pop_size = pop_size, dim=dim, max_iters=max_iters)

        # GOPSO default
        self.c1        = 1.49618
        self.c2        = 1.49618
        self.w         = 0.72984
        self.jump_rate = 0.3

    def init_states(self):
        randomPositions = np.random.random((self.pop_size, self.dim))*(self.xMax - self.xMin) + self.xMin

        # Compute opposite population and concatenate with original population
        initPopulation = pd.DataFrame(np.concatenate((randomPositions,
                                                      getOppositeNumber(randomPositions, self.xMin, self.xMax, k='random'))))
        initPopulation  = self.set_state(initPopulation)

        # Keep only the fittest from set [pop, opposite_pop]
        self.population = initPopulation.sort_values("Fitness", ascending=True, inplace=False).iloc[:self.pop_size, :]
        self.population = self.population.reset_index(drop=True)

        self.velocity     = pd.DataFrame(np.zeros((self.pop_size, self.dim)))

        # Update best positions and costs found per particle
        self.previousBest = self.population.copy()

        # Update Global best position and cost
        self.update_global_best()
        return self.population

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

        oppositePop = getOppositeNumber(self.population.iloc[:, :-1], infLimit, supLimit)
        oppositePop = self.set_state(oppositePop, substitute='random')

        expandedPopulation = pd.concat([oppositePop, self.population], axis=0, ignore_index=True)

        # Keep only the fittest from set [pop, opposite_pop]
        self.population = expandedPopulation.sort_values("Fitness", ascending=True, inplace=False).reset_index(drop=True)
        self.population = self.population.iloc[:self.pop_size, :].reset_index(drop=True)

        return self.population

    def mutate_best_vector(self):
        # Mutation with Cauchy Standard distribution
        # mutatedPos = np.array([(self.bestVector.iloc[:-1] + np.random.standard_cauchy(size=self.dim))])
        mutatedPos = self.bestVector.iloc[0, :-1] + np.random.standard_cauchy(size=self.dim)

        mutatedBestVector = pd.DataFrame(np.expand_dims(mutatedPos, axis=0))
        mutatedBestVector = self.set_state(mutatedBestVector)

        if mutatedBestVector.loc[0, "Fitness"] <= self.bestVector.loc[0, "Fitness"]:
            self.bestVector = mutatedBestVector
        return self.bestVector

    def generation(self):
        # Perform Generation jumping
        if np.random.rand() < self.jump_rate:
            self.generation_jumping()

        # Standard PSO generation loop
        super().generation()

        # Mutate best vector
        self.mutate_best_vector()
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

class DifferentialEvolution(EvolutionaryAlgorithm):    
    def __init__(self, dim=2, func_id=1, pop_size=30, 
                 crossover = 'binomial', prob_cr=.9, pop_corpus='real', 
                 opposition=False, mutation='best', lambda_mutation=1, n_diff=1, 
                F = [.8], substitute = 'random', x_min=-100, x_max=100):
        # Initialize superclass EvolutionaryAlgorithm
        super().__init__(dim=dim, pop_size=pop_size)
        self.xMin       = x_min       # Search space limits
        self.xMax       = x_max       #
        self.func_id    = func_id

        # Parameters
        self.param_F    = F   # Mutation parameter F. If len(F) == n_diff, each 
                              # F is applied differently to each diff mutation
        self.prob_cr = prob_cr   # Crossover probability
        self.opposition = opposition
        self.pop_corpus = pop_corpus # ['real', 'integer', 'binary']
        self.mutation = mutation # ['best', 'rand', 'mixed' ]
        self.lambda_mutation = lambda_mutation # float in (0, 1)
        self.n_diff = n_diff # [1, 2, 3, ...] (usually 1 or 2)
        self.substitute = substitute # ['random', 'edge', 'none'] what to do to outside specimen
        self.crossover = crossover # ['bin', 'exp']
        self.generations = 0        
        self.maxFitnessEvals = None
        self.fitnessEvalsHist = [0]        
        self.population = None
        self.trialPopulation = None
        self.mutatedPopulation = None        
        self.fitness_clusters = None # To be used by CDE

        # Fitness Function definition
        if self.func_id < 23:
            self.problem = pg.problem(pg.cec2014(prob_id=self.func_id, dim=self.dim))
        else:
            raise NotImplementedError("f_{:2d} not yet implemented".format(self.func_id))
            
        # Initialize population DataFrame and compute initial Fitness
        self.init_states()

    def __str__(self):
        return "DE/" + self.mutation + "/" + str(self.n_diff) + "/" + self.crossover[:3]
        
    def init_states(self):
        ''' Randomly initialize states and sigma values following a uniform initialization
            between limits xMax and xMin. Assign state and fitness values to self.population.

            population is a DataFrame with one row per individual, one column per
            dimension and one extra for fitness values.

            Arguments: None

            Returns: self.population, DataFrame with dimensions (population size) by (dimension + 1).
        '''
        
        specimen = Population(dimension=self.dim, lowerLimit=self.xMin, upperLimit=self.xMax, 
                       initialPopulation=self.pop_size, method=self.pop_corpus, opposition=self.opposition).create()        
        initPopulation = pd.DataFrame(specimen)
        #self.population = self.set_state(initPopulation)
        initPopulation  = self.set_state(initPopulation)

        # Keep only the fittest from set [pop, opposite_pop]
        self.population = initPopulation.sort_values("Fitness", ascending=True, inplace=False).iloc[:self.pop_size, :]
        self.population = self.population.reset_index(drop=True)
        return self.population.copy()
    
    def get_fitness(self, specimen):
        ''' Wrapper that returns fitness value for state input specimen and increments
            number of fitness evaluations.

            Argument: specimen. State vector of length (dim).

            Returns : Fitness for given input state as evaluated by target function.
        '''
        self.fitnessEvals +=1        
        return self.problem.fitness(specimen)[0]
    
    def set_state(self, newPopulation, fitness_clusters=None):
        ''' Function to attribute new values to a population DataFrame.
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
                pd.DataFrame(np.where(logicArray, 
                                      Population(dimension=self.dim, lowerLimit=self.xMin, 
                                                 upperLimit=self.xMax, initialPopulation=self.pop_size, 
                                                 method=self.pop_corpus, opposition=self.opposition).create(),
                                      newPopulation
                                     )),
            'none':
                pd.DataFrame(np.where(logicArray, np.NaN, newPopulation)),
        #     'edge':
        #         newPopulation.clip(lower=self.xMin).clip(upper=self.xMax)
        # 
        }
        updatedPopulation = choices.get(self.substitute, KeyError("Please select a valid substitute key")).copy()

        # Compute new Fitness            
        if fitness_clusters is None:
            fitness = updatedPopulation.apply(self.get_fitness, axis=1).copy()
        else:
            # If self.fitness_clusters > 0, then the updated generation is clustered and we only make evaluations for the cluster's 
            # center. The final fitness for each specimen is the fitness of its corresponding cluster center            
            kmeans = KMeans(n_clusters=fitness_clusters, n_jobs=-1, verbose=False)
            kmeans.fit(updatedPopulation)

            # Getting cluster for each specimen 
            df_clusters = pd.DataFrame(kmeans.labels_, columns=['index'])

            # Finding cluster center fitness
            cluster_fitness = pd.DataFrame(kmeans.cluster_centers_).apply(self.get_fitness, axis=1).copy()
            df_fitness = pd.DataFrame(cluster_fitness, columns=['Fitness'])
            df_fitness.reset_index(drop = False, inplace=True)

            # Mapping each specimen to the cluster's center fitness
            fitness = pd.merge(df_clusters, df_fitness, on='index', how='inner')['Fitness']
        
        updatedPopulation = updatedPopulation.assign(Fitness=fitness)

        if updatedPopulation.isna().any().any():            
            print("\nWarning: NA values found in population\n")
            print(updatedPopulation.drop(labels="Fitness", axis=1).isna().sum(axis=1))
            input()

        return updatedPopulation.copy()
    
    def mutate_differential(self, indexes, method='best'):
        """
            method: ['best', 'rand', 'mixed']
            n_diff: [1, 2, ...] (usually 1 or 2)
            u = lambda*best + (1-lambda)*rand + F*(x_1-x_2) + F*(x3-x4) + ...
        """
        
        best_vector = self.population.iloc[self.population.idxmin(axis=0)['Fitness'], :-1]
        base_vector = self.population.iloc[indexes[0], :-1] 
        
        if self.mutation == 'rand':
            lambda_mutation = 0
        elif self.mutation == 'best':
            lambda_mutation = 1
        else:
            lambda_mutation = self.lambda_mutation
        
        result_vector = lambda_mutation*best_vector + (1-lambda_mutation)*base_vector
        
        # Repeating param_F to contain number of diffs
        param_F = np.repeat(self.param_F[0], self.n_diff) if len(self.param_F) != self.n_diff else self.param_F        
        
        count = 1        
        for i in np.arange(self.n_diff):            
            random_vector1 = self.population.iloc[indexes[count], :-1]
            random_vector2 = self.population.iloc[indexes[count+1], :-1]
            result_vector += param_F[i]*(random_vector1 - random_vector2)
            count += 2

        return result_vector
    
    def mutate(self):
        ''' For each specimenIndex in self.population, chooses randomly selected indexes 
        from [0...pop_size-1] which are different from specimentIndex and applies 
        self.mutate_differential to each Triplet        
        '''
        def make_index_list(index):
            indexList = list(range(0, self.pop_size))
            indexList.remove(index)
            return np.random.choice(indexList, size=1+2*self.n_diff, replace=False)

        # Create list of random indexes
        randomIndexes = np.array(list(map(make_index_list, list(range(0, self.pop_size)))))

        # Mutate every specimen using scheme passed to mutationScheme        
        self.mutatedPopulation = pd.DataFrame(np.apply_along_axis(self.mutate_differential, axis=1, arr=randomIndexes))

        return self.mutatedPopulation

    def perform_crossover(self, fitness_clusters=None):
        ''' Performs binomial or exponential crossover. Compose a trial vector based on each mutated
            vector and its corresponding parent. 
            Returns self.trialPopulation DataFrame with updated fitness column
        '''        
        if self.crossover == 'exp':
            # j: starting index 
            L = np.random.geometric(self.prob_cr, 1)
            j = np.random.randint(0, self.dim)            
            
            newPopulation = pd.DataFrame(                
                np.where(np.isin(np.arange(self.dim), np.arange(j, j+L+1, 1)%self.dim),                         
                self.mutatedPopulation, 
                self.population.iloc[:, :-1])
            ) 

        else: # if self.crossover == 'bin':
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
            #   In other words, Keep old values only if (randomArray > cross_rate) AND (randomK != column) 
            newPopulation = pd.DataFrame(np.where(np.logical_or(np.less_equal(randomArray, self.prob_cr),
                                                                 np.equal(randomK, maskArray)),
                                                  self.mutatedPopulation, self.population.iloc[:, :-1]))

        # Compute new fitness values and treat exceptions
        self.trialPopulation = self.set_state(newPopulation, fitness_clusters)
            
        return self.trialPopulation

    def select_survivor(self):
        ''' DE survivor selection. A mutant vector is carried over to the next
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

    def generate(self):        
        self.mutate()
        self.perform_crossover()
        self.select_survivor()
        self.generations += 1
        
        return self.population
    
    def optimize(self, target, max_f_evals='auto', max_generations=None, target_error=10e-8, verbose=True):
        """
            returns errorHist and fitnessHist which are (self.generations, self.pop_size)
        
        """
        if max_f_evals == 'auto':
            max_f_evals = 10000*self.dim

        # if max_f_evals is not None:
        self.setMaxFitnessEvals(max_f_evals)

        # Initialize variables
        fitnessHist = pd.DataFrame()
        errorHist = pd.DataFrame()        
        
        while True:        
            # Stop Conditions
            if (max_f_evals is not None and self.fitnessEvals + self.generations > max_f_evals):
                if (self.fitness_clusters is None):
                    print ('Optimization ended due to max fitness evals (max = {}, curr = {})'.format(self.maxFitnessEvals, self.fitnessEvals))
                    break
                elif (self.fitnessEvals + self.fitness_clusters > max_f_evals):
                    print ('Clustered Optimization ended due to max fitness evals (max = {}, curr = {})'.format(self.maxFitnessEvals, self.fitnessEvals))
                    break
                
            # Setting next generation
            self.generate()

            # Save error and fitness history
            fitnessHist = fitnessHist.append(self.population["Fitness"])              
            errorHist   = fitnessHist.copy() - target
            errorHist   = errorHist.apply(np.abs).reset_index(drop=True)

            bestError   = errorHist.iloc[-1,:].min()
            
            if (self.fitnessEvals > max_f_evals) or (bestError <= target_error) or (max_generations is not None and self.generations > max_generations):
                break            

        fitnessHist.reset_index(drop=True, inplace=True)
        fitnessHist.index.name = 'generation' 
        errorHist.index.name = 'generation'

        # Mean and Best fitness values of last generation
        lastMeanFit = fitnessHist.iloc[self.generations-1, :].mean()
        lastBestFit = fitnessHist.iloc[self.generations-1, :].min()

        if verbose is True:
            print("\n#Generations:\t{}".format(self.generations))
            print("#FitnessEvals:\t{}".format(self.fitnessEvals))
            print("Mean Fitness:\t{:.9f}".format(lastMeanFit))
            print("Best Fitness:\t{:.9f}\nSolution:\t{:.49}\nDiff:\t\t{:.9f}".format(lastBestFit, target, abs(target-lastBestFit)))

        return errorHist, fitnessHist

    def setMaxFitnessEvals(self, maxFitnessEvals):
        self.maxFitnessEvals = maxFitnessEvals

class ClusteredDifferentialEvolution(DifferentialEvolution):
    def __init__(self, dim=2, func_id=1, pop_size=100, crossover='binomial', prob_cr=.9, pop_corpus='real',
        mutation='best', lambda_mutation=1, n_diff=1, F=[.8], substitute='random', cluster_rate=.1, fitness_clusters=10):
        # Initialize superclass DifferentialEvolution
        super().__init__(dim=dim, func_id=func_id, pop_size=pop_size, 
                 crossover=crossover, prob_cr=prob_cr, pop_corpus=pop_corpus, 
                 opposition=False, mutation=mutation, lambda_mutation=lambda_mutation, n_diff=n_diff, 
                F=F, substitute=substitute)

        # Child class parameters        
        self.cluster_rate = cluster_rate   # Fitness cluster probability
        self.fitness_clusters = fitness_clusters # (int or None) whether to calculate fitness only to population kmeans center
        
    def __str__(self):
        return "CDE/" + self.mutation + "/" + str(self.n_diff) + "/" + self.crossover[:3]

    def generate(self):
        init_fitness_evals = self.fitnessEvals

        self.mutate()        
        # Deciding whether to set state by clustering or not
        if np.random.random() <= self.cluster_rate and self.fitnessEvals > 0.5*self.maxFitnessEvals:                        
            self.perform_crossover(self.fitness_clusters)
        else:
            self.perform_crossover()
        self.select_survivor()

        self.fitnessEvalsHist.append(self.fitnessEvals-init_fitness_evals)
        self.generations += 1
        return self.population

class OppositionDifferentialEvolution(DifferentialEvolution):
    def __init__(self, dim=2, func_id=1, pop_size=100, crossover='binomial', prob_cr=.9, pop_corpus='real',
        mutation='best', lambda_mutation=1, n_diff=1, F=[.8], substitute='random', jump_rate=.3):
        # Initialize superclass DifferentialEvolution
        self.xMin       = -100       # Search space limits
        self.xMax       =  100       #        

        super().__init__(dim=dim, func_id=func_id, pop_size=pop_size, 
                 crossover=crossover, prob_cr=prob_cr, pop_corpus=pop_corpus, 
                 opposition=True, mutation=mutation, lambda_mutation=lambda_mutation, n_diff=n_diff, 
                F=F, substitute=substitute)

        # Parameters        
        self.jump_rate = jump_rate   # Generation jumping probability

    def __str__(self):
        return "ODE/" + self.mutation + "/" + str(self.n_diff) + "/" + self.crossover[:3]

    def generation_jumping(self):
        ''' Compute an opposite population to self.population and keep fittest specimens.
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

        oppositePop = getOppositeNumber(self.population.iloc[:, :-1], infLimit, supLimit)
        oppositePop = self.set_state(oppositePop)

        expandedPopulation = pd.concat([oppositePop, self.population], axis=0, ignore_index=True)

        # Keep only the fittest from set [pop, opposite_pop]
        self.population = expandedPopulation.sort_values("Fitness", ascending=True, inplace=False).reset_index(drop=True)
        self.population = self.population.iloc[:self.pop_size, :].reset_index(drop=True)
        
        return self.population

    def generate(self):
        init_fitness_evals = self.fitnessEvals

        self.mutate()
        self.perform_crossover()
        self.select_survivor()

        if np.random.random() <= self.jump_rate:            
            self.generation_jumping()

        self.fitnessEvalsHist.append(self.fitnessEvals-init_fitness_evals)
        self.generations += 1
        return self.population

class QuasiOppositionDifferentialEvolution(OppositionDifferentialEvolution):
    def __init__(self, dim=2, func_id=1, pop_size=100, crossover='binomial', prob_cr=.9, pop_corpus='real',
        mutation='best', lambda_mutation=1, n_diff=1, F=[.8], substitute='random', jump_rate=.3):
        # Initialize superclass DifferentialEvolution        
        super().__init__(dim=dim, func_id=func_id, pop_size=pop_size, crossover=crossover, 
            prob_cr=prob_cr, pop_corpus=pop_corpus, mutation=mutation, lambda_mutation=lambda_mutation, 
            n_diff=n_diff, F=F, substitute=substitute, jump_rate=jump_rate)

    def __str__(self):
        return "QODE/" + self.mutation + "/" + str(self.n_diff) + "/" + self.crossover[:3]

    def generation_jumping(self):
        ''' Compute an opposite population to self.population and keep fittest specimens.
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

        oppositePop = getQuasiOppositeNumber(self.population.iloc[:, :-1], infLimit, supLimit)
        oppositePop = self.set_state(oppositePop)

        expandedPopulation = pd.concat([oppositePop, self.population], axis=0, ignore_index=True)

        # Keep only the fittest from set [pop, opposite_pop]
        self.population = expandedPopulation.sort_values("Fitness", ascending=True, inplace=False).reset_index(drop=True)
        self.population = self.population.iloc[:self.pop_size, :].reset_index(drop=True)
        
        return self.population

class ClusteredOppositionDifferentialEvolution(OppositionDifferentialEvolution):
    def __init__(self, dim=2, func_id=1, pop_size=100, crossover='binomial', prob_cr=.9, pop_corpus='real',
        mutation='best', lambda_mutation=1, n_diff=1, F=[.8], substitute='random', 
        cluster_rate=.1, fitness_clusters=10, jump_rate=.3):
        # Initialize superclass OppositionDifferentialEvolution
        super().__init__(dim=dim, func_id=func_id, pop_size=pop_size, 
                 crossover=crossover, prob_cr=prob_cr, pop_corpus=pop_corpus, 
                 mutation=mutation, lambda_mutation=lambda_mutation, n_diff=n_diff, 
                F=F, substitute=substitute, jump_rate=jump_rate)

        # Child class parameters        
        self.cluster_rate = cluster_rate   # Fitness cluster probability
        self.fitness_clusters = fitness_clusters # (int or None) whether to calculate fitness only to population kmeans center
        
    def __str__(self):
        return "CODE/" + self.mutation + "/" + str(self.n_diff) + "/" + self.crossover[:3]

    def generate(self):
        init_fitness_evals = self.fitnessEvals

        self.mutate()        
        # Deciding whether to set state by clustering or not
        if np.random.random() <= self.cluster_rate:                        
            self.perform_crossover(self.fitness_clusters)
        else:
            self.perform_crossover()
        self.select_survivor()

        if np.random.random() <= self.jump_rate:            
            self.generation_jumping()

        self.fitnessEvalsHist.append(self.fitnessEvals-init_fitness_evals)
        self.generations += 1
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
    # Standard arguments
    dim = 10
    func_id = 1
    pop_size = 30
    maxGenerations = None #1000
    maxEvals = 'auto'
    solution = get_solution(func_id, dim)

    # Differential Evolution-based Algorithms
    mutation = 'best'
    n_diff = 1
    crossover = 'binonial'
    lambda_mutation = 1

    # Cluster-based Differential Evolution
    fitness_clusters = 10
    cluster_rate = 0.05

    # Opposition-based Differential Evolution
    jump_rate = .3
    opposition = False

    # Differential Evolution
    # alg = DifferentialEvolution(dim=dim, func_id=func_id, pop_size=pop_size, crossover=crossover, 
    #     opposition=opposition, mutation=mutation, lambda_mutation=lambda_mutation)

    # Clustered Differential Evolution
    alg = ClusteredDifferentialEvolution(dim=dim, func_id=func_id, pop_size=pop_size, crossover=crossover, 
        mutation=mutation, lambda_mutation=lambda_mutation, 
        fitness_clusters=fitness_clusters, cluster_rate=cluster_rate)

    # Opposite Differential Evolution
    # alg = OppositionDifferentialEvolution(dim=dim, func_id=func_id, pop_size=pop_size, crossover=crossover, 
    #     mutation=mutation, lambda_mutation=lambda_mutation, jump_rate=jump_rate, n_diff=n_diff)

    # Quasi-Opposite Differential Evolution
    # alg = QuasiOppositionDifferentialEvolution(dim=dim, func_id=func_id, pop_size=pop_size, crossover=crossover, 
    #     mutation=mutation, lambda_mutation=lambda_mutation, jump_rate=jump_rate, n_diff=n_diff)

    # Clustered Opposite Differential Evolution
    # alg = ClusteredOppositionDifferentialEvolution(dim=dim, func_id=func_id, pop_size=pop_size, crossover=crossover, 
    #     mutation=mutation, lambda_mutation=lambda_mutation, jump_rate=jump_rate, n_diff=n_diff, fitness_clusters=fitness_clusters, cluster_rate=cluster_rate)

    print ("Testing " + str(alg) + " on func_id " + str(func_id) + " for " + str(dim) + " dimensions")    
    print ("Mutation: ", alg.mutation, 'Lambda', alg.lambda_mutation, 'CR', alg.crossover)
    
    errorHist, fitnessHist = alg.optimize(target=solution, max_f_evals=maxEvals)

    print ("Fitness shape", fitnessHist.shape)
    print ("ErrorHist shape", errorHist.shape)    
    print ("Final population\n", alg.population.head())