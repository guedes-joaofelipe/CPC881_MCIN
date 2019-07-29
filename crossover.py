import numpy as np
import population as p

def makeCrossOver(parent1, parent2, method='onePointCrossOver', verbose=False):
    if method.lower() == 'onepointcrossover':
        return onePointCrossOver(parent1, parent2, verbose=verbose)
    else: 
        raise NotImplementedError

def onePointCrossOver(parent1, parent2, verbose=False):    
    position = np.random.randint(1, parent1.shape[0])
    print ('Onepoint Crossover position: ', position) if verbose else None
    child1 = np.append(parent1[0:position], parent2[position:])
    child2 = np.append(parent2[0:position], parent1[position:])
    return child1, child2

def main():
    parent1 = p.initializePopulationReal(5, 0 ,5, 1)[0]
    parent2 =  p.initializePopulationReal(5, 0 ,5, 1)[0]
    child1, child2 = onePointCrossOver(parent1, parent2, True)
    print ('parent1: {}\nparent2: {}\nchild1: {}\nchild2: {}'.format(parent1, parent2, child1, child2))

if __name__ == "__main__":
    main()        
            