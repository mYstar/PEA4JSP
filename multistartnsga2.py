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
rank = comm.Get_rank()

# read parameters
generations, pop_size, f_model, _, _,\
        mut_prob, mut_eta, xover_prob, xover_eta = params.get()

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

# ---  main GA loop  ---
for _ in range(generations):

    # -- execute genetic operators --
    # selection
    emo.assignCrowdingDist(population)
    offspring = tools.selTournamentDCD(population, len(population))

    # crossover and mutation
    offspring = algorithms.varAnd(
        offspring,
        toolbox,
        cxpb=xover_prob,
        mutpb=1.0)  # is taken care of by mutation operator

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
    population = toolbox.select(
        offspring,
        len(population))

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
