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
import output


class NSGA2(object):

    """Contains all the needed functions to perform an optimization with the
    Master-Slave NSGA-II."""

    def __init__(self, modelfile):
        """Creates the JSPModel

        :modelfile: The file to read the model from.

        """
        self.model = JspModel(modelfile)
        self.evaluator = JspEvaluator(self.model)

    def optimize(self, term_m, term_v, population, mut_pb, mut_eta,
                 xover_pb, xover_eta):
        """Performs the Optimisation via Master-Slave NSGA-II.

        :term_m: the termination method (see: params.get())
        :term_v: the termination value (see: params.get())
        :population: the size of the population to use
        :mut_pb: mutation probability per genome
        :mut_eta: mutation spread (1.0: high spread; 4.0: low spread)
        :xover_pb: crossover probability
        :xover_eta: crossover spread (2.0: high spread; 5.0: low spread)

        :returns: None

        """
        # creation of individuals
        fitness_size = self.evaluator.metrics_count()
        weights = tuple([-1 for _ in range(fitness_size)])
        creator.create("FitnessMin", base.Fitness, weights=weights)
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

        toolbox.register("mate", operators.crossover, eta=xover_eta)
        toolbox.register("mutate", operators.mutation,
                         indpb=mut_pb, eta=mut_eta)
        toolbox.register("select", tools.selNSGA2)

        population = toolbox.population(n=pop)
        fits = map(
            lambda x: operators.calc_fitness(x, self.evaluator),
            population)

        for fit, i_pop in zip(fits, population):
            i_pop.fitness.values = fit

        gen = 0
        while not operators.termination(term_m, term_v, gen, population):
            gen += 1
            # selection of mates
            emo.assignCrowdingDist(population)
            offspring = tools.selTournamentDCD(population, len(population))

            # crossover and mutation
            offspring = algorithms.varAnd(
                offspring,
                toolbox,
                cxpb=xover_pb,
                mutpb=1.0)  # is handled already by the mutation operator

            # calculate new fitness
            fits = map(
                lambda x: operators.calc_fitness(x, self.evaluator),
                offspring)

            for fit, i_off in zip(fits, offspring):
                i_off.fitness.values = fit

            # select individuals for the next generation
            offspring.extend(population)
            population = toolbox.select(
                offspring,
                len(population))

        return population

if __name__ == '__main__':
    term_m, term_v, pop, f_out, f_model, _, _, mut_pb,\
        mut_eta, xover_pb, xover_eta = params.get()

    alg = NSGA2(f_model)
    population = alg.optimize(term_m, term_v, pop, mut_pb,
                              mut_eta, xover_pb, xover_eta)

    makespan, twt, flow, setup, load, wip =\
        output.get_min_metric(population)

    with open(f_out, "a") as myfile:
        myfile.write("{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n".format(
            makespan, twt, flow, setup, load, wip))
