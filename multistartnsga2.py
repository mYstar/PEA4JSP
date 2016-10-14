""" Implementation of the Master-Slave NSGA-II Genetic Algorithm using JSPEval
to evaluate the individuals. The fitness function is evaluated in parallel by
the Slave processors.
"""
import random
import time
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
rank = comm.Get_rank()
size = comm.Get_size()
print(size)

# read parameters
term_m, term_v, pop_size, f_out, f_model, _, _,\
        mut_prob, mut_eta, xover_prob, xover_eta = params.get()

# start multiple runs
start = time.time()

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
gen = 0
terminate = False
term_reqs = []
for node in range(size):
    term_reqs.append(comm.irecv(source=node, tag=0))

while not terminate:
    gen += 1

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
        lambda x: operators.calc_fitness(x, evaluator),
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

    terminate = operators.termination(term_m, term_v, gen, population)

    # send a termination signal to all others
    # needed for makespan termination
    if terminate:
        print('rank: {} termination, sending signal'.format(rank))
        for node in range(size):
            comm.isend(True, node, tag=0)

    # test for termination of others
    _, node_term, _ = MPI.Request.testany(term_reqs)
    if node_term:
        print('rank: {}, termination signal received'.format(rank))
    terminate = terminate | node_term

# ---  process results ---

# collect results
sol_values = np.empty([pop_size, model.solution_length()])
fit_values = np.empty([pop_size, fitness_size])
for i, ind in zip(range(pop_size), population):
    sol_values[i] = ind.get_values()
    fit_values[i] = ind.fitness.values

sol_all = None
fit_all = None
if rank == 0:
    sol_all = np.empty([pop_size * size, model.solution_length()])
    fit_all = np.empty([pop_size * size, fitness_size])

comm.Gather(sol_values, sol_all, root=0)
comm.Gather(fit_values, fit_all, root=0)

if rank == 0:
    all_pop = toolbox.population(n=pop_size * size)
    for i, ind in zip(range(pop_size * size), all_pop):
        ind.set_values(sol_all[i])
        ind.fitness.values = fit_all[i]

    duration = time.time() - start

    output.write_pareto_front(all_pop, f_out)

    with open('{}.time'.format(f_out), 'a') as myfile:
        myfile.write('{}\n'.format(duration))
