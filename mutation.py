import numpy as np
import population as p

#Binary Representation
def binaryMutation(parent, probability):
    mutate = lambda x: 1-x if (np.random.rand() < probability) else x
    return np.vectorize(mutate)(parent)

#Integer Representation
def integerRandomResetting(parent, probability, lower, upper):
    mutate = lambda x: np.random.randint(lower, upper) if (np.random.rand() < probability) else x
    return np.vectorize(mutate)(parent)

#Validar se depois da soma os valores passaram os limites da populacao
def creepMutation(parent, probability, lower, upper):    
    mutate = lambda x: x + np.random.randint(lower, upper) if (np.random.rand() < probability) else x
    return np.vectorize(mutate)(parent)

#Float Representation
def uniformMutation(parent, probability, lower, upper):
    arr_random = np.random.rand(len(parent))    
    arr_substitute = (upper - lower)*(arr_random) + lower    
    return np.where(arr_random < probability, arr_substitute, parent)


def main():
    arr = np.array([[1, 0], [0, 1]])
    probability = -1
    # print ('Binary mutation (prob={}): \n{} \n>> \n{}'.format(probability, arr, binaryMutation(arr, probability)))
    # print ('Integer resetting (prob={}): \n{} \n>> \n{}'.format(probability, arr, integerRandomResetting(arr, probability, 0, 5)))
    population = p.Population(initialPopulation=4, lowerLimit=-100, upperLimit=100, dimension=4).create()
    print(population)

    print (uniformMutation(population[0], probability, -100, 100))

if __name__ == "__main__":
    main()        
            
