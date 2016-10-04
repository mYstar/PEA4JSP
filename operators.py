from deap import tools
import time


def calc_fitness(individual, evaluator):
    """ Function for fitness calculation.

    :solution: the solution vector of an individual
    :evaluator: the JspEval to calculate the metrics with
    """

    assign = evaluator.build_machine_assignment(individual)
    sched = evaluator.execute_schedule(assign)
    metrics = evaluator.get_metrics(assign, sched)

    return metrics


def init_individual(ind_cls, model, value_func):
    """ Initialises a JspSolution

    :ind_cls: class to init (has to be drawn from JspSolution
    :model: the JspModel to use
    :value_func: a function to create the value array
    """
    values = value_func()
    return ind_cls(model, values)


def crossover(ind1, ind2, eta):
    """Performs a 2-Point Crossover on Individuals.

    :ind1: first parent
    :ind2: second parent
    :returns: two offsprings as a tuple

    """
    # get the genomes
    genome1 = ind1.get_values()
    genome2 = ind2.get_values()

    # perform the crossover only on the genomes
    genome1, genome2 = tools.cxSimulatedBinaryBounded(
        genome1,
        genome2,
        eta,  # eta_c 2.0 == wide search; 5.0 narrow search
        0.0,  # lower bound
        1.0)  # upper bound

    # change the values of the individuals
    ind1.set_values(genome1)
    ind2.set_values(genome2)

    return (ind1, ind2)


def mutation(individual, indpb, eta):
    """Mutates an individual using polynomial bounded mutation.

    :individual: the individual to mutate
    :indpb: the probability for a mutation to occur
    :returns: the (altered) individual

    """
    gen = tools.mutPolynomialBounded(
        individual.get_values(),
        eta,  # eta_m 1.0 == high spread; 4.0 low spread
        0.0,  # lower bound
        1.0,  # upper bound
        indpb)
    individual.set_values(gen[0])

    return (individual,)


def termination(method, value, generation, population):
    """Checks for the termination criterion. Knows 3 different termination methods:
        - after a number of generations
        - after an amount of wall clock time
        - when a defined makespan is met

    :method: a string that determines the method ('generations', 'time' or
             'makespan')
    :value: the comparison value to use
    :generation: the current generation
    :population: the population to check for makespan
    :returns: a boolean value, if the termination criterion is met

    """
    if method == 'generations':
        return value <= generation
    if method == 'time':
        return value < time.time()
    if method == 'makespan':
        for ind in population:
            if value > ind.fitness.values[0]:
                return True
        return False

    return True
