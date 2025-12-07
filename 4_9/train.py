from deap import base, algorithms
from deap import creator
from deap import tools
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

inf = 100
D = ((0, 3, 1, 3, inf, inf),
    (3, 0, 4, inf, inf, inf),
    (1, 4, 0, inf, 7, 5),
    (3, inf, inf, 0, inf, 2),
    (inf, inf, 7, inf, 0, 4),
    (inf, inf, 5, 2, 4, 0))

startV = 0
LENGTH_D = len(D)
LENGTH_CHROM = len(D) * len(D[0])

POPULATION_SIZE = 500
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 30

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("randomOrder", random.sample, range(LENGTH_D), LENGTH_D)
toolbox.register("individualCreator", tools.initRepeat,
creator.Individual, toolbox.randomOrder, LENGTH_D)
toolbox.register("populationCreator", tools.initRepeat, list,
toolbox.individualCreator)

population = toolbox.populationCreator(n=POPULATION_SIZE)

def dikstryFitness(individual):
    s = 0
    for n, path in enumerate(individual):
        path = path[:path.index(n)+1]
        si = startV
        for j in path:
            s += D[si][j]
            si = j
    return s, # кортеж

def cxOrdered(ind1, ind2):
    for p1, p2 in zip(ind1, ind2):
        tools.cxOrdered(p1, p2)
    return ind1, ind2

def mutShuffleIndexes(individual, indpb):
    for ind in individual:
        tools.mutShuffleIndexes(ind, indpb)
    return individual,

toolbox.register("evaluate", dikstryFitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", cxOrdered)
toolbox.register("mutate", mutShuffleIndexes, indpb=1.0/LENGTH_CHROM/10)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)

population, logbook = algorithms.eaSimple(population, toolbox, 
                                          cxpb=P_CROSSOVER/LENGTH_D,
                                          mutpb=P_MUTATION/LENGTH_D,
                                          ngen=MAX_GENERATIONS,
                                          stats=stats,
                                          verbose=True)

maxFitnessValues, meanFitnessValues = logbook.select("min", "avg")

plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()

HALL_OF_FAME_SIZE = 1
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

population, logbook = algorithms.eaSimple(population, toolbox,
                                          cxpb=P_CROSSOVER/LENGTH_D,
                                          mutpb=P_MUTATION/LENGTH_D,
                                          ngen=MAX_GENERATIONS,
                                          halloffame=hof,
                                          stats=stats,
                                          verbose=True)

best = hof.items[0]
print(best)

def show_graph(ax, best):
    #точки произвольно, в задаче 6 вершин
    vertex = ((0, 1), (1, 1), (0.5, 0.8), (0.1, 0.5), (0.8, 0.2), (0.4, 0))
    vx = [v[0] for v in vertex]
    vy = [v[1] for v in vertex]
    ax.add_line(Line2D((vertex[0][0], vertex[1][0]), (vertex[0][1],
    vertex[1][1]), color='#aca'))
    ax.add_line(Line2D((vertex[0][0], vertex[2][0]), (vertex[0][1],
    vertex[2][1]), color='#aca'))
    ax.add_line(Line2D((vertex[0][0], vertex[3][0]), (vertex[0][1],
    vertex[3][1]), color='#aca'))
    ax.add_line(Line2D((vertex[1][0], vertex[2][0]), (vertex[1][1],
    vertex[2][1]), color='#aca'))
    ax.add_line(Line2D((vertex[2][0], vertex[5][0]), (vertex[2][1],
    vertex[5][1]), color='#aca'))
    ax.add_line(Line2D((vertex[2][0], vertex[4][0]), (vertex[2][1],
    vertex[4][1]), color='#aca'))
    ax.add_line(Line2D((vertex[3][0], vertex[5][0]), (vertex[3][1],
    vertex[5][1]), color='#aca'))
    ax.add_line(Line2D((vertex[4][0], vertex[5][0]), (vertex[4][1],
    vertex[5][1]), color='#aca'))
    startV = 0
    for i, v in enumerate(best):
        if i == 0:
            continue
        prev = startV
        v = v[:v.index(i)+1]
        for j in v:
            ax.add_line(Line2D((vertex[prev][0], vertex[j][0]),
                               (vertex[prev][1], vertex[j][1]), color='r'))
            prev = j
    ax.plot(vx, vy, 'ob', markersize=20)

fig, ax = plt.subplots()
show_graph(ax, best)
plt.show()