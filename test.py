import pygmo    as pg

prob = pg.problem(pg.schwefel(30))

alg = pg.algorithm(pg.sade(gen=100))

arch = pg.archipelago(16, algo=alg, prob=prob, pop_size=20)

arch.evolve(10)

arch.wait()

results = [isl.get_population().champion_f[0] for isl in arch]

print(results)
