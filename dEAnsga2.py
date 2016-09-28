""" Implementation of the Master-Slave NSGA-II Genetic Algorithm using JSPEval
to evaluate the individuals. The fitness function is evaluated in parallel by
the Slave processors.
"""
import random
import numpy as np
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
size = comm.Get_size()
topology = params.calculate_topology(size)
cart = MPI.Intracomm(comm).Create_cart(
        [topology[0], topology[1]],
        [True, True])
rank = cart.Get_rank()

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
reqs = []
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
    fits = map(
        lambda x: operators.calc_fitness(
            JspSolution(model, x.values),
            evaluator),
        offspring)

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
        print('rank: {}, gen: {}, solutions sent'.format(rank, gen))
        # select solutions for migration
        migrants = toolbox.select(population, migration_size)
        sol_send = np.empty(
            [migration_size, model.solution_length()],
            dtype=np.float32)
        for i, ind in zip(range(migration_size), migrants):
            sol_send[i] = ind.values

        # send best solutions to neighbors
        for req in reqs:
            req.Free()
        reqs = []
        sol_recv = np.empty(
            [len(neighbors), migration_size, model.solution_length()],
            dtype=np.float32)
        for i, nb in zip(range(len(neighbors)), neighbors):
            cart.Isend(sol_send, nb, tag=gen)
            reqs.append(cart.Irecv(sol_recv[i], source=nb, tag=gen))
        ready = False

    # receive migrants
    if not ready:
        ready, _ = MPI.Request.testall(reqs)
        if ready:
            reqs = []
            print('rank: {}, gen: {}, solutions received'.format(rank, gen))
            # flatten list
            sol_recv = [values for n_list in sol_recv for values in n_list]

            # build solutions
            migrants = []
            for values in sol_recv:
                migrants.append(creator.Individual(model, values))

            # calculate fitness
            migfit = map(
                lambda x: operators.calc_fitness(
                    JspSolution(model, x.values), evaluator),
                migrants)
            for fit, mig in zip(migfit, migrants):
                mig.fitness.values = fit

            # integrate into population
            population = toolbox.select(
                population,
                len(population)-len(migrants))
            population.extend(migrants)


# ---  process results ---

# collect results
all_pop = comm.gather(population, root=0)

# output
if rank == 0:
    all_pop = sum(all_pop, [])
    makespan, twt, flow, setup, load, wip =\
        output.get_min_metric(all_pop)
    print('rank: {}'.format(rank))
    print('best makespan: {}'.format(makespan))
    print('best twt: {}'.format(twt))
    print('best flow: {}'.format(flow))
    print('best setup: {}'.format(setup))
    print('best load: {}'.format(load))
    print('best wip: {}'.format(wip))
else:
    makespan, twt, flow, setup, load, wip =\
        output.get_min_metric(population)
    print('rank: {}'.format(rank))
    print('best makespan: {}'.format(makespan))
    print('best twt: {}'.format(twt))
    print('best flow: {}'.format(flow))
    print('best setup: {}'.format(setup))
    print('best load: {}'.format(load))
    print('best wip: {}'.format(wip))
