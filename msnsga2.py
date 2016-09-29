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
rank = comm.Get_rank()

if rank == 0:
    # read parameters
    generations, pop_size, f_model, _, _,\
            mut_prob, mut_eta, xover_prob, xover_eta = params.get()
    pop_size *= size

    # -- setup algorithm --

    # init evaluator
    model = JspModel(f_model)
    evaluator = JspEvaluator(model)

    # init GA
    fitness_size = evaluator.metrics_count()
    weights = tuple([-1 for _ in range(fitness_size)])
    creator.create("FitnessMin", base.Fitness, weights=weights)
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
    toolbox.register("mate", operators.crossover, eta=xover_eta)
    toolbox.register("mutate", operators.mutation, indpb=mut_prob, eta=mut_eta)
    toolbox.register("select", tools.selNSGA2)

    # init first population
    population = toolbox.population(n=pop_size)
    fits = map(lambda x:  operators.calc_fitness(x, evaluator), population)
    for fit, i_pop in zip(fits, population):
        i_pop.fitness.values = fit
else:
    evaluator = None
    generations = None
    pop_size = None

# ---  broadcast parameters and model  ---
evaluator = comm.bcast(evaluator, root=0)
generations = comm.bcast(generations, root=0)
pop_size = comm.bcast(pop_size, root=0)

# ---  main GA loop  ---
root_values = None
for _ in range(generations):

    # -- execute genetic operators --
    if rank == 0:
        # selection
        emo.assignCrowdingDist(population)
        offspring = tools.selTournamentDCD(population, len(population))

        # crossover and mutation
        offspring = algorithms.varAnd(
            offspring,
            toolbox,
            cxpb=xover_prob,
            mutpb=1.0)  # is handled by mutation operator itself

        # prepare individuals for scatter
        root_values = np.empty([pop_size, model.solution_length()])
        for i, ind in zip(range(len(offspring)), offspring):
            root_values[i] = ind.values
        root_values = np.reshape(
            root_values,
            (size, int(pop_size/size), model.solution_length()))

    # -- PARALLEL: calculate fitness --

    # scatter values over cores
    remote_values = \
        np.empty([int(pop_size/size), evaluator.model.solution_length()])
    comm.Scatter(root_values, remote_values, root=0)

    # calculation
    newfit = map(
        lambda x: operators.calc_fitness(
            JspSolution(evaluator.model, x),
            evaluator),
        remote_values)

    # gather results
    remote_fits = np.array(list(newfit))
    fits = None
    if rank == 0:
        fits = np.empty([pop_size, fitness_size])
    comm.Gather(remote_fits, fits, root=0)

    # -- select next population --
    if rank == 0:
        # assign fitness
        fits = np.reshape(fits, (pop_size, fitness_size))
        for fit, i_off in zip(fits, offspring):
            i_off.fitness.values = fit

        # selection
        offspring.extend(population)
        population = toolbox.select(
            offspring,
            len(population))

# ---  process results ---
if rank == 0:
    makespan, twt, flow, setup, load, wip =\
        output.get_min_metric(population)

    print('best makespan: {}'.format(makespan))
    print('best twt: {}'.format(twt))
    print('best flow: {}'.format(flow))
    print('best setup: {}'.format(setup))
    print('best load: {}'.format(load))
    print('best wip: {}'.format(wip))
