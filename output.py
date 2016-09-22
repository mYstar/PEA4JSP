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
    :returns: a set with the individuals

    """
    pareto_front = tools.sortNondominated(population, 500, True)
    uniq = set()
    for ind in pareto_front[0]:
        uniq.add(ind.fitness.values)

    return uniq
