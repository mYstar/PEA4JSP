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


generations, pop_size, f_model = params.get()
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
generations = 10
pop_size *= size

# broadcast the model to all cores
if rank == 0:
    model = JspModel(f_model)
    evaluator = JspEvaluator(model)
else:
    evaluator = None

evaluator = comm.bcast(evaluator, root=0)

print('rank: {}, solution_length={}'.format(
    rank,
    evaluator.model.solution_length()))

if rank == 0:
    # creation of individuals
    creator.create(
        "FitnessMin",
        base.Fitness,
        weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
        )
    creator.create("Individual", JspSolution, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("values",  # alias
                     tools.initRepeat,  # fill container by repetition
                     list,  # container type
                     random.random,  # fill function
                     model.solution_length())  # number of repetitions

    toolbox.register("individual",  # alias
                     operators.init_individual,  # generator function
                     creator.Individual,  # individual class
                     model,  # model to use
                     toolbox.values)  # value generator
    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual)

    toolbox.register("mate", operators.crossover)
    toolbox.register("mutate", operators.mutation, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)

    population = toolbox.population(n=pop_size)

    fits = map(lambda x:  operators.calc_fitness(x, evaluator), population)
    for fit, i_pop in zip(fits, population):
        i_pop.fitness.values = (fit[0], fit[1], fit[2], fit[3], fit[4], fit[5])
else:
    toolbox = None

off_values = None
for _ in range(generations):
    if rank == 0:
        # selection of mates
        emo.assignCrowdingDist(population)
        offspring = tools.selTournamentDCD(population, len(population))

        # crossover and mutation
        offspring = algorithms.varAnd(
            offspring,
            toolbox,
            cxpb=0.5,
            mutpb=0.1)

        # prepare individuals for scatter
        off_values = np.empty([pop_size, model.solution_length()])
        for i, ind in zip(range(len(offspring)), offspring):
            off_values[i] = ind.values
        off_values = np.reshape(
            off_values,
            (size, int(pop_size/size), model.solution_length()))

    # calculate new fitness
    remote_values = np.empty(
        [int(pop_size/size), evaluator.model.solution_length()])
    comm.Scatter(off_values, remote_values, root=0)
    remote_fits = np.array(
        list(map(
            lambda x: operators.calc_fitness(
                JspSolution(evaluator.model, x),
                evaluator),
            remote_values)),
        dtype=np.float32)

    fits = None
    if rank == 0:
        fits = np.empty(
            [pop_size, 6],
            dtype=np.float32)

    comm.Gather(remote_fits, fits, root=0)

    if rank == 0:
        fits = np.reshape(fits, (pop_size, 6))
        for fit, i_off in zip(fits, offspring):
            i_off.fitness.values = (fit[0], fit[1], fit[2],
                                    fit[3], fit[4], fit[5])

        # select individuals for the next generation
        offspring.extend(population)
        population = toolbox.select(
            offspring,
            len(population))

if rank == 0:
    pareto_front = tools.sortNondominated(population, 500, True)
    uniq = set()
    for ind in pareto_front[0]:
        uniq.add(ind.fitness.values)
    for ind in uniq:
        print(ind)

    mo = JspModel('JSPEval/xml/example.xml')
    ev = JspEvaluator(mo)
    for ind in pareto_front[0]:
        if ind.fitness.values[0] == 40.0:  # select fastest solution
            assign = ev.build_machine_assignment(ind)
            sched = ev.execute_schedule(assign)
            print(assign)
            print(sched)
