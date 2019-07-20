import numpy as np
from utils import getOppositeNumber

class Population:

    def __init__(self, dimension, lowerLimit, upperLimit, initialPopulation, method = 'real', opposition = False):
        self.dimension = dimension
        self.lowerLimit = lowerLimit
        self.upperLimit = upperLimit
        self.initialPopulation = initialPopulation
        self.method = method.lower()
        self.opposition = opposition

    def getDimension(self):
        return self.dimension

    def setDimentison(self, dimension):
        self.dimension = dimension

    def getLowerLimit(self):
        return self.lowerLimit

    def setLowerLimit(self, lowerLimit):
        self.lowerLimit = lowerLimit

    def getUpperLimit(self):
        return self.upperLimit

    def setUpperLimit(self, upperLimit):
        self.upperLimit = upperLimit

    def getInitialPopulation(self):
        return self.initialPopulation

    def setInitialPopulation(self, initialPopulation):
        self.initialPopulation = initialPopulation

    def create(self):
        if not self.opposition:
            pop_size = self.initialPopulation
        else: 
            pop_size = int(self.initialPopulation/2)+1 if self.initialPopulation%2 == 1 else int(self.initialPopulation/2)

        if self.method == 'real':
            population = (self.getUpperLimit() - self.getLowerLimit())*(np.random.rand(pop_size, self.dimension)) + self.lowerLimit
        elif self.method == 'integer':
            population = np.random.randint(self.getLowerLimit(), self.getUpperLimit(), size=(pop_size, self.dimension))
        else: 
            population = np.random.randint(0, 2, size=(pop_size, self.dimension))

        if self.opposition:
            oppositePopulation = getOppositeNumber(population, self.getLowerLimit(), self.getUpperLimit(), k=1)            
            population = np.append(population, oppositePopulation, axis=0)
            if self.initialPopulation%2 == 1:
                # Removing extra specimen due to append doubling
                population = population[:-1]            
        self.population = population

        return population

    
if __name__ == "__main__":  
    pop = Population(dimension=4, lowerLimit=-100, upperLimit=100, initialPopulation=4, method='real', opposition=True).create()
    print("Initial Population:\n", pop)
