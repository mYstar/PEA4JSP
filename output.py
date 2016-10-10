from deap import tools


def get_min_metric(population):
    """Prints the min value for every metric.
    :population: the population to process
    :returns: a tuple with the values
    """
    pf = get_pareto_front(population)

    makespan = min(pf, key=lambda item: item[0])[0]
    twt = min(pf, key=lambda item: item[1])[1]
    flow = min(pf, key=lambda item: item[2])[2]
    setup = min(pf, key=lambda item: item[3])[3]
    load = min(pf, key=lambda item: item[4])[4]
    wip = min(pf, key=lambda item: item[5])[5]

    return (makespan, twt, flow, setup, load, wip)


def get_pareto_front(population):
    """Calculates the first pareto-front.

    :population: the population to process
    :returns: a list with the individuals and a set with their fitness values

    """
    pareto_front = tools.sortNondominated(population, 500, True)
    fitness = set()
    solutions = list()
    fitness_size = 0
    for ind in pareto_front[0]:
        fitness.add(ind.fitness.values)
        # add only the first solution with the same fitness
        if fitness_size < len(fitness):
            solutions.append(ind.get_values())
            fitness_size = len(fitness)

    return solutions, fitness


def write_pareto_front(population, p_file):
    """ Calculates the paretofront of the population and appends it to the
    file.

    :population: the population to process
    :p_file: the filename to write to
    :returns: None

    """
    solutions, fitness = get_pareto_front(population)

    with open(p_file + '.fit', 'a') as myfile:
        for solution in fitness:
            myfile.write(', '.join(str(x) for x in solution))
            myfile.write('\n')

    with open(p_file + '.sol', 'a') as myfile:
        for solution in solutions:
            myfile.write(', '.join(str(x) for x in solution))
            myfile.write('\n')
