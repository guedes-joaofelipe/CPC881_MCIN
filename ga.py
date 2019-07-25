import pandas as pd
import numpy as np

from evolution import EvolutionaryAlgorithm
import mutation as m
import crossover as c
import selection as s

class GeneticAlgorithm(EvolutionaryAlgorithm):
    def __init__(self, dim, func_id, pop_size, prob_mutation=.2, mutation='uniform', 
        prob_cr=.8, crossover='onePointCrossOver', elitism=1, fitness_clusters=None, pop_corpus='real'):

        # Initialize superclass EvolutionaryAlgorithm
        super().__init__(dim=dim, func_id=func_id, pop_size=pop_size, prob_cr=prob_cr, crossover=crossover, 
            mutation=mutation, prob_mutation=None, pop_corpus=pop_corpus, 
            fitness_clusters=fitness_clusters)

        self.elitism = elitism
        
    def __str__(self):
        return "GA/" + self.mutation + "/" + str(self.elitism) + "/" + self.crossover[:3]
        
    def perform_crossover(self, parent1, parent2, verbose=False):
        return c.makeCrossOver(parent1, parent2, self.crossover, verbose=verbose)

    def mutate(self, individual):

        if self.mutation == 'uniform':
            individual = m.uniformMutation(individual,self.prob_mutation,self.xMin, self.xMax)
        else:
            raise NotImplementedError

        return individual

    def select_elite(self, parents):
        return parents[:self.elitism]

    def generate(self, selection, parents_number):
        # returns new population

        # Selecting parents        
        parents = s.makeSelection(self.population.iloc[:, :-1]), 
                                self.population['Fitness'], 
                                method=selection, N=parents_number, ascending=False)

        # # Making crossover
        next_generation = list()
        # Combining every parent 2x2
        for combination in list(itertools.product(parents, parents)):
            p1, p2 = combination[0], combination[1]
            if not np.array_equal(p1,p2) and (np.random.rand() < self.prob_crossover):            
                child1, child2 = c.makeCrossOver(p1, p2, self.crossover, verbose=False)
                next_generation.append(child1)
                next_generation.append(child2)
        
        # # Mutation
        # for index, individual in enumerate(next_generation):
        #     if (np.random.rand() < prob_mutation):
        #         individual = m.uniformMutation(individual,0.2,lowerLimit, upperLimit)
        #         next_generation[index] = individual

        # # Elitism
        # for best_parent in parents[:elitism]:
        #     next_generation.append(best_parent)
            
        # # Evaluating next generation
        # generation_eval = np.array([])
        # for individual in next_generation:
        #     generation_eval = np.append(generation_eval, e.cecFunction(function_number, individual, dim))

        # next_generation = np.array(next_generation)    
        
        # generations['individuals'].append(next_generation)
        # generations['evaluations'].append(generation_eval)


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

        if max_f_evals is not None:
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
                    print ('Clustered Optimization ended due to max fitness evals (max = {}, curr = {})'.format(self.max_f_evals, self.fitnessEvals))
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
            print("Mean Fitness:\t{:.4f}".format(lastMeanFit))
            print("Best Fitness:\t{:.4f}\nSolution:\t{:.4f}\nDiff:\t\t{:.4f}".format(lastBestFit, target, abs(target-lastBestFit)))

        return errorHist, fitnessHist

