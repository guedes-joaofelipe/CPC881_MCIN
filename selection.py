import numpy as np

def makeSelection(population, method='tournament', N=10, ascending=False):
    if method.lower() == 'tournament':
        return tournament(population, N=N, ascending=ascending)
    else:
        raise NotImplementedError

def tournament(population, N=10, ascending=False):
    """ selects the parents for the next generation
    :param generation (np.array): current generation
    :param score (np.array): score for each individual of current generation
    :N (int): number of parents to be selected
    :ascending (boolean): if False, the N highest scores are selected. 
        Otherwise, the N lowest scores are selected    
    """
    return population.sort_values(by=['Fitness'], ascending=ascending).head(N).reset_index(drop = True).copy()


def main():
    generation = np.array([1, 2, 3, 4])
    score = [.4, .6, .8, .2]
    # refazer
    # print(topNScore(generation, score, N=10, ascending=False))

if __name__ == "__main__":
    main()    