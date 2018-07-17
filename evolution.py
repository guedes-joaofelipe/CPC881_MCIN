import numpy    as np
import pandas   as pd
import pygmo    as pg
from copy       import copy

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
        self.mutateProb      = 0.01
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
            # print("\nState Perturbation: \n{}".format(newSigma*stateNoise))
            # print("\nNew Sigma: \n{}".format(newSigma))
            newState[:self.dim] = x[:self.dim] + newSigma*stateNoise
            newState[self.dim:] = newSigma
            # print("\nNew X:\n", newState)

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
        # numParticipants = 2*self.pop_size   # Selection with parents and children

        tourneyPop = pd.concat([self.childrenPopulation, self.population], axis=0, ignore_index=True)
        numParticipants = tourneyPop.shape[0]

        tourneyPop["Score"] = np.zeros(numParticipants)

        for i in range(numParticipants):
            opponents = np.random.randint(0, numParticipants, size=numPlays)

            currList = np.ones(numPlays)*tourneyPop.loc[i, "Fitness"]
            # print("Shape curr list: ", currList.shape)
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
