""" Implementation of the Master-Slave NSGA-II Genetic Algorithm using JSPEval
to evaluate the individuals. The fitness function is evaluated in parallel by
the Slave processors.
"""
import random
from deap import creator, base, tools, algorithms
from deap110 import emo
from JSPEval.jspsolution import JspSolution
from JSPEval.jspmodel import JspModel
from JSPEval.jspeval import JspEvaluator
import params
import operators


class NSGA2(object):

    """Contains all the needed functions to perform an optimization with the
    Master-Slave NSGA-II."""

    def __init__(self, modelfile):
        """Creates the JSPModel

        :modelfile: The file to read the model from.

        """
        self.model = JspModel(modelfile)
        self.evaluator = JspEvaluator(self.model)

    def optimize(self, generations, population):
        """Performs the Optimisation via Master-Slave NSGA-II.

        :modelfile: the xml file to read the model from
        :generations: the number of generations to perform
        :population: the size of the population to use
        :returns: None

        """
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
                         self.model.solution_length())  # number of repetitions

        toolbox.register("individual",  # alias
                         operators.init_individual,  # generator function
                         creator.Individual,  # individual class
                         self.model,    # fix argument
                         toolbox.values)  # value generator
        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual)

        toolbox.register(
            "evaluate",
            operators.calc_fitness,
            evaluator=self.evaluator)
        toolbox.register("mate", operators.crossover)
        toolbox.register("mutate", operators.mutation, indpb=0.05)
        toolbox.register("select", tools.selNSGA2)

        population = toolbox.population(n=pop)
        fits = toolbox.map(toolbox.evaluate, population)

        for fit, i_pop in zip(fits, population):
            i_pop.fitness.values = fit

        for _ in range(generations):
            # selection of mates
            emo.assignCrowdingDist(population)
            offspring = tools.selTournamentDCD(population, len(population))

            # crossover and mutation
            offspring = algorithms.varAnd(
                offspring,
                toolbox,
                cxpb=0.5,
                mutpb=0.1)

            # calculate new fitness
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, i_off in zip(fits, offspring):
                i_off.fitness.values = fit

            # select individuals for the next generation
            offspring.extend(population)
            population = toolbox.select(
                offspring,
                len(population))

        return tools.sortNondominated(population, 500, True)

if __name__ == '__main__':
    gen, pop, f_model = params.get()

    alg = NSGA2(f_model)
    pareto_front = alg.optimize(gen, pop)

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
