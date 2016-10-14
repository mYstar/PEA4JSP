import sys
import time
import getopt
import math


def get():
    """Parses the commandline parameters for:
        - termination method
        - termination value
        - population size
        - modelfile for optimization
        - migration interval
        - migration size
        - mutation probability
        - mutation eta (1.0: high spread; 4.0: low spread)
        - crossover probability
        - crossover eta (2.0: high spread; 5.0: low spread)
        - output file
        - modelfile

    :returns: a tuple (term_meth, term_val, pop, outputfile, modelfile,
            mig_int, mig_size, mut_prob, mut_eta, xover_prob, xover_eta)
    """
    usage_string = "usage: python3 <algorithm>.py --term-method\
<generations|time|makespan> -t <term_value> -p <population per core>\
--mi <migration intervall> --ms <migration size> --mp <mutation probability>\
--me <mutation eta>\ xp <crossover probability> --xe <crossover eta>\
-o <outputfile> <modelfile>"

    # read the given parameters
    try:
        options, files = getopt.getopt(
                sys.argv[1:],
                "ht:p:o:",
                ["help", "population=", "mi=", "ms=", "mp=", "output",
                 "me=", "xp=", "xe=", "term-method=", "term-value="])
    except getopt.GetoptError:
        print(usage_string)
        sys.exit(1)

    term_value = 10
    term_method = 'generations'
    pop = 100
    migr_i = 5
    migr_s = 5
    mut_pb = 0.01
    mut_eta = 1.5
    xover_pb = 1.0
    xover_eta = 5.0
    output = 'experiment'

    f_model = './JSPEval/xml/example.xml'
    for opt, arg in options:
        if opt in ("-h", "--help"):
            print(usage_string)
            sys.exit()
        elif opt in ('-t', '--term-value'):
            term_value = float(arg)
        elif opt in ('-o', '--output'):
            output = arg
        elif opt in ('--term-method'):
            term_method = arg
            if term_method not in ('generations', 'time', 'makespan'):
                print('termination method unknown using: generations')
                term_method = 'generations'
        elif opt in ('-p', '--population'):
            pop = int(arg)
        elif opt == "--mi":
            migr_i = int(arg)
        elif opt == "--ms":
            migr_s = int(arg)
        elif opt == "--mp":
            mut_pb = float(arg)
        elif opt == "--me":
            mut_eta = float(arg)
        elif opt == "--xp":
            xover_pb = float(arg)
        elif opt == "--xe":
            xover_eta = float(arg)

    # calculate the endtime
    if term_method == 'time':
        term_value = term_value * 60 + time.time()

    if len(files) > 0:
        f_model = files[0]

    if not pop % 4 == 0:
        raise ValueError('The population size has to be divisible by 4 \
for the selection to work.')

    if not migr_s <= pop:
        raise ValueError('migration size cannot exceed population.')

    if not migr_i > 0:
        raise ValueError('migration interval has to be positive.')

    print('Algorithm using:\ntermination method: {}\ntermination value: {}\
    population: {}\nmodelfile: {}'.format(
        term_method,
        term_value,
        pop,
        f_model))
    print('migration size: {}\nmigration interval: {}'
          .format(migr_s, migr_i))
    print('mutation prob: {}\nmutation eta: {}'
          .format(mut_pb, mut_eta))
    print('crossover prob: {}\ncrossover eta: {}'
          .format(xover_pb, xover_eta))

    return (term_method, term_value, pop, output, f_model, migr_i,
            migr_s, mut_pb, mut_eta, xover_pb, xover_eta)


def calculate_topology(nodes):
    """Calculates the optimal (most connections) 2D topology for a number of nodes.

    :nodes: the number of nodes to calculate the topology for
    :returns: a tuple (width, height) of the resulting topology

    """
    root = int(math.sqrt(nodes))

    for i in range(root, 1, -1):
        if nodes % i == 0:
            return (i, int(nodes/i))

    return (1, nodes)
