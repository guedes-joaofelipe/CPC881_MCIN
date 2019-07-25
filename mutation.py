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
    for i in range(len(parent)):
        if (np.random.rand() < probability):
            parent[i] = (upper - lower)*(np.random.rand()) + lower
    return parent        

def main():
    arr = np.array([[1, 0], [0, 1]])
    probability = .8
    print ('Binary mutation (prob={}): \n{} \n>> \n{}'.format(probability, arr, binaryMutation(arr, probability)))
    print ('Integer resetting (prob={}): \n{} \n>> \n{}'.format(probability, arr, integerRandomResetting(arr, probability, 0, 5)))
    print(uniformMutation(p.initializePopulationReal(5, 0 ,5, 1), 0.5, 0, 5))

if __name__ == "__main__":
    main()        
            
