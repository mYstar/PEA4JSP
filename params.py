import sys
import getopt
import math


def get():
    """Parses the commandline parameters for:
        - number of generations
        - population size
        - modelfile for optimization
        - migration interval
        - migration size
        - mutation probability
        - mutation eta (1.0: high spread; 4.0: low spread)
        - crossover probability
        - crossover eta (2.0: high spread; 5.0: low spread)

    :returns: a tuple (gen, pop, file, mig_int, mig_size,
                       mut_prob, mut_eta, xover_prob, xover_eta)
    """
    usage_string = "usage: python3 <algorithm>.py -g \
<generations> -p <population per core> --mi <migration intervall>\
--ms <migration size> --mp <mutation probability> --me <mutation eta>\
--xp <crossover probability> --xe <crossover eta> <modelfile>"

    # read the given parameters
    try:
        options, files = getopt.getopt(
                sys.argv[1:],
                "hg:p:",
                ["help", "generations=", "population=", "mi=",
                 "ms=", "mp=", "me=", "xp=", "xe="])
    except getopt.GetoptError:
        print(usage_string)
        sys.exit(1)

    gen = 10
    pop = 100
    migr_i = 5
    migr_s = 5
    mut_pb = 0.05
    mut_eta = 1.0
    xover_pb = 1.0
    xover_eta = 2.0

    f_model = './JSPEval/xml/example.xml'
    for opt, arg in options:
        if opt in ("-h", "--help"):
            print(usage_string)
            sys.exit()
        elif opt in ('-g', '--generations'):
            gen = int(arg)
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

    if len(files) > 0:
        f_model = files[0]

    if not pop % 4 == 0:
        raise ValueError('The population size has to be divisible by 4 \
for the selection to work.')

    if not migr_s <= pop:
        raise ValueError('migration size cannot exceed population.')

    if not migr_i > 0:
        raise ValueError('migration interval has to be positive.')

    print('Algorithm using:\ngenerations: {}\
    population: {}\nmodelfile: {}'.format(gen, pop, f_model))
    print('migration size: {}\nmigration interval: {}'
          .format(migr_s, migr_i))
    print('mutation prob: {}\nmutation eta: {}'
          .format(mut_pb, mut_eta))
    print('crossover prob: {}\ncrossover eta: {}'
          .format(xover_pb, xover_eta))

    return (gen, pop, f_model, migr_i, migr_s,
            mut_pb, mut_eta, xover_pb, xover_eta)


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
