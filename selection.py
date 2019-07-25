import numpy as np

def makeSelection(generation, score, method='topNScore', N=10, ascending=False):
    if method.lower() == 'topnscore':
        return topNScore(generation, score, N=N, ascending=ascending)
    else:
        raise NotImplementedError

def topNScore(generation, score, N=10, ascending=False):
    """ selects the parents for the next generation
    :param generation (np.array): current generation
    :param score (np.array): score for each individual of current generation
    :N (int): number of parents to be selected
    :ascending (boolean): if False, the N highest scores are selected. 
        Otherwise, the N lowest scores are selected    
    """
    generation = np.array(generation)     
    index_parents = np.argsort(-np.array(score))[:N] if ascending else np.argsort(np.array(score))[:N]
    parents = generation[index_parents]    
    return parents

def main():
    generation = np.array([1, 2, 3, 4])
    score = [.4, .6, .8, .2]
    print(topNScore(generation, score, N=10, ascending=False))

if __name__ == "__main__":
    main()    