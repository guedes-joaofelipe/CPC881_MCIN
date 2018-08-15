import numpy    as np
import pandas   as pd
import pygmo    as pg
from copy       import copy

from utils      import opposite_number

class EvolutionaryAlgorithm:
    def __init__(self, dim=2, pop_size=10):
        print("DE Init setting pop size ", pop_size)
        self.dim            = dim
        self.pop_size       = pop_size

        self.fitnessEvals   = 0


class DifferentialEvolution(EvolutionaryAlgorithm):
    def __init__(self, dim=2, func_id=1, pop_size=30):
        # Initialize superclass EvolutionaryAlgorithm
        print("DE Init passing pop size ", pop_size)
        super().__init__(dim=dim, pop_size=pop_size)
        self.xMin       = -100       # Search space limits
        self.xMax       =  100       #

        self.func_id    = func_id

        # Parameters
        self.param_F    = 0.9   # Mutation parameter F
        self.cross_rate = 0.2   # Crossover probability

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
        print("DE Init")
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
                pd.DataFrame(np.where(logicArray, np.random.random()*(self.xMax - self.xMin) + self.xMin, newPopulation)),
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

    def mutation_best_1(self, specimen, index, param_F=0.1):
        '''
            DE/best/1 mutation scheme
            Every specimen produces a mutated/donor vector each generation.

            Arguments:
                specimen: Target vector of shape (1 by dim+1), including Fitness column
                index   : Target vector index.
        '''
        # Select two new specimens, different from current specimen
        randomNum1 = index
        while randomNum1 == index:
            randomNum1 = np.random.randint(0, self.pop_size)

        randomNum2 = index
        while randomNum2 == index:
            randomNum2 = np.random.randint(0, self.pop_size)

        # Sanity check
        if (randomNum1 == index) or (randomNum2 == index):
            raise ValueError("Mutation index equal target index")
            return -1

        bestSpecimen  = self.population.sort_values("Fitness", ascending=True, inplace=False).iloc[0, :-1]
        randSpecimen1 = self.population.iloc[randomNum1, :-1]
        randSpecimen2 = self.population.iloc[randomNum2, :-1]

        mutatedSpecimen = bestSpecimen + param_F*(randSpecimen1 - randSpecimen2)

        return mutatedSpecimen

    def mutation(self, mutation_scheme):
        indexList     = list(range(0, self.pop_size))
        parameterList = self.param_F*np.ones(self.pop_size)

        # Mutate every specimen using scheme passed to mutationScheme
        # mutationScheme = self.mutation_best_1
        self.mutatedPopulation = pd.DataFrame(list(map(mutation_scheme, self.population.T, indexList, parameterList))).reset_index(drop=True)

        # self.mutatedPopulation = self.set_state(newPopulation, substitute='edge')

        return self.mutatedPopulation

    def crossover_binomial(self, mutatedPopulation):
        '''
            DE binomial crossover. Compose a trial vector based on each mutated
            vector and its corresponding parent. The new vector is composed following:

                u_ij = v_ij,  if j == K or rand(0,1) <= cross_rate
                       x_ij,  else

            The trial vector will use the mutated value if probability is within
            cross_rate or its columns is randomly selected by K = randint(0, dim).

            Returns self.trialPopulation DataFrame with updated fitness column
        '''
        # Create random number arrays
        randomArray = np.random.rand(self.mutatedPopulation.shape[0], self.mutatedPopulation.shape[1])
        randomK     = np.random.randint(0, self.dim, size=self.pop_size)
        maskArray   = np.arange(self.mutatedPopulation.shape[1])

        # Reshape and tile arrays
        randomK.shape   = (self.pop_size, 1)
        randomK         = np.tile(randomK, (1, self.dim))

        maskArray.shape = (1, self.mutatedPopulation.shape[1])
        maskArray       = np.tile(maskArray, (self.pop_size, 1))

        # print(randomArray.shape)
        # print(randomK.shape)
        # print(maskArray.shape)
        # input()

        newPopulation = self.population.drop(labels='Fitness', axis=1)

        # Substitute new values for randomArray smaller than Crossover rate
        newPopulation = newPopulation.where(np.less_equal(randomArray, self.cross_rate),
                                            other=self.mutatedPopulation)

        # Substitute new values for K == columns, guaranteeing at least one substitution
        newPopulation = newPopulation.where(np.not_equal(randomK, maskArray),
                                            other=self.mutatedPopulation)

        # Compute new fitness values and treat exceptions
        self.trialPopulation = self.set_state(newPopulation, substitute='edge')

        return self.trialPopulation

    def survivor_selection(self, trialPopulation):
        '''
            DE survivor selection. A mutant vector is carried over to the next
            generation if its fitness is better or equal than its parent's.

                x_i(t+1) = u_i(t), if f(u_i(t)) <= f(x_i(t))
                           x_i(t), else

            Returns updated self.population DataFrame
        '''

        # Logic array to compare new and previous fitness values
        logicArray = np.less_equal(trialPopulation['Fitness'], self.population['Fitness']).values
        logicArray.shape = (self.pop_size, 1)
        logicArray = np.tile(logicArray, (1, self.dim+1))

        # Keep mutated specimens only if fitness improves over its parents
        newPopulation = pd.DataFrame(np.where(logicArray, trialPopulation, self.population))

        newPopulation.columns = self.population.columns # Horrible hack to recover Fitness columns name

        self.population = newPopulation.copy()

        return self.population

    def generation(self):
        self.mutation(self.mutation_best_1)
        self.crossover_binomial(self.mutatedPopulation)
        self.survivor_selection(self.trialPopulation)

        return self.population

class OppositionDifferentialEvolution(DifferentialEvolution):
    def __init__(self, dim=2, func_id=1, pop_size=100):
        # Initialize superclass DifferentialEvolution
        print("ODE Init passing pop size ", pop_size)
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

            Returns: self.population, DataFrame with dimensions (population size)
            by (2*dimension + 1).
        '''
        print("OP DE init")
        # Initialize population uniformly over search space and compute opposite population
        pop = np.random.random((self.pop_size, self.dim))*(self.xMax - self.xMin) + self.xMin
        oppositePop = opposite_number(pop, self.xMin, self.xMax)

        initPopulation = pd.DataFrame(np.concatenate((pop, oppositePop)))

        initPopulation  = self.set_state(initPopulation, substitute='random')

        # Keep only the fittest from set [pop, opposite_pop]
        self.population = initPopulation.sort_values("Fitness", ascending=True, inplace=False).iloc[:self.pop_size, :]
        self.population = self.population.reset_index(drop=True)
        return self.population.copy()

    def mutation_rand_1(self, specimen=None, index=None, param_F=0.1):
        '''
            DE/best/1 mutation scheme
            Every specimen produces a mutated/donor vector each generation.

            Arguments:
                specimen, index are mantained for compatibility.
                param_F: F parameter. Must be a positive real number.
        '''
        # Select random target specimen
        index = np.random.randint(0, self.pop_size)
        specimen = self.population.iloc[index, :]

        # Select two new specimens, different from current specimen
        randomNum1 = index
        while randomNum1 == index:
            randomNum1 = np.random.randint(0, self.pop_size)

        randomNum2 = index
        while randomNum2 == index:
            randomNum2 = np.random.randint(0, self.pop_size)

        # Sanity check
        if (randomNum1 == index) or (randomNum2 == index):
            raise ValueError("Mutation index equal target index")
            return -1

        bestSpecimen  = self.population.sort_values("Fitness", ascending=True, inplace=False).iloc[0, :-1]
        randSpecimen1 = self.population.iloc[randomNum1, :-1]
        randSpecimen2 = self.population.iloc[randomNum2, :-1]

        mutatedSpecimen = bestSpecimen + param_F*(randSpecimen1 - randSpecimen2)

        return mutatedSpecimen

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
        infLimit.shape   = (1, self.dim)
        supLimit.shape   = (1, self.dim)

        infLimit         = np.tile(infLimit, (self.pop_size, 1))
        supLimit         = np.tile(supLimit, (self.pop_size, 1))

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
