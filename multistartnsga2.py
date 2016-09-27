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
for _ in range(generations):

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
    population = toolbox.select(
        offspring,
        len(population))

# ---  process results ---
makespan, twt, flow, setup, load, wip =\
    output.get_min_metric(population)

print('rank: {}'.format(rank))
print('best makespan: {}'.format(makespan))
print('best twt: {}'.format(twt))
print('best flow: {}'.format(flow))
print('best setup: {}'.format(setup))
print('best load: {}'.format(load))
print('best wip: {}'.format(wip))
