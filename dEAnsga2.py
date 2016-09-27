""" Implementation of the Master-Slave NSGA-II Genetic Algorithm using JSPEval
to evaluate the individuals. The fitness function is evaluated in parallel by
the Slave processors.
"""
import random
from mpi4py import MPI
from deap import creator, base, tools, algorithms
from deap110 import emo
from JSPEval.jspsolution import JspSolution
from JSPEval.jspmodel import JspModel
from JSPEval.jspeval import JspEvaluator
import params
import operators
import output

# ---  Setup  ---

# MPI environment
comm = MPI.COMM_WORLD
cart = MPI.Intracomm(comm).Create_cart([2, 2], [True, True])
rank = cart.Get_rank()
size = cart.Get_size()

# calculate neighbors
coord = cart.Get_coords(rank)
neighbor_coords = [[coord[0]-1, coord[1]],
                   [coord[0], coord[1]-1],
                   [coord[0]+1, coord[1]],
                   [coord[0], coord[1]+1]]
neighbors = set(map(cart.Get_cart_rank, neighbor_coords))
print('rank: {}, neighbors: {}'.format(rank, neighbors))

# read parameters
generations, pop_size, f_model = params.get()

# -- setup algorithm --

# init evaluator
model = JspModel(f_model)
evaluator = JspEvaluator(model)

# init GA
creator.create("FitnessMin", base.Fitness,
               weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0))
creator.create("Individual", JspSolution, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("values",
                 tools.initRepeat,
                 list,
                 random.random,
                 model.solution_length())
toolbox.register("individual",  # alias
                 operators.init_individual,  # generator function
                 creator.Individual,  # individual class
                 model,  # model to use
                 toolbox.values)  # value generator
toolbox.register("population",
                 tools.initRepeat,
                 list,
                 toolbox.individual)
toolbox.register("mate", operators.crossover)
toolbox.register("mutate", operators.mutation, indpb=0.05)
toolbox.register("select", tools.selNSGA2)

# init first population
population = toolbox.population(n=pop_size)
fits = map(lambda x:  operators.calc_fitness(x, evaluator), population)
for fit, i_pop in zip(fits, population):
    i_pop.fitness.values = fit

# ---  main GA loop  ---
for gen in range(generations):

    # -- execute genetic operators --
    # selection
    emo.assignCrowdingDist(population)
    offspring = tools.selTournamentDCD(population, len(population))

    # crossover and mutation
    offspring = algorithms.varAnd(
        offspring,
        toolbox,
        cxpb=0.5,
        mutpb=0.1)

    # fitness calculation
    fits = map(lambda x: operators.calc_fitness(x, evaluator), offspring)

    # -- select next population --
    # assign fitness
    for fit, i_off in zip(fits, offspring):
        i_off.fitness.values = fit

    # selection
    offspring.extend(population)
    population = toolbox.select(offspring, len(population))

    # --- migration ---

    migration_interval = 5
    migration_size = 5
    # migrate own solutions
    if gen % 5 == 0:
        print('rank: {}, gen: {}, solutions sent')
        # select solutions for migration
        sol_send = toolbox.select(offspring, migration_size)
        # send best solutions to neighbors
        reqs = []
        for nb in neighbors:
            cart.isend(sol_send, nb, tag=gen)
            reqs.append(cart.irecv(source=nb, tag=gen))
        ready = False

    # receive migrants
    if not ready:
        ready, sol_recv = MPI.Request.testall(reqs)
        if ready:
            print('rank: {}, gen: {}, solutions received')
            # flatten list (uses overloaded '+' operator)
            sol_recv = sum(sol_recv, [])
            print('solutions: {}'.format(sol_recv))
            population = toolbox.select(
                population,
                len(population)-len(sol_recv))
            population.extend(sol_recv)


# ---  process results ---
makespan, twt, flow, setup, load, wip =\
    output.get_min_metric(population)

# collect results

# generate pareto front

# output
print('rank: {}'.format(rank))
print('best makespan: {}'.format(makespan))
print('best twt: {}'.format(twt))
print('best flow: {}'.format(flow))
print('best setup: {}'.format(setup))
print('best load: {}'.format(load))
print('best wip: {}'.format(wip))
