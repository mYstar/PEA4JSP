import sys
import getopt


def get():
    """Parses the commandline parameters for:
        - number of generations
        - population size
        - modelfile for optimization

    :returns: a tuple (gen, pop, file)
    """
    usage_string = "usage: python3 <algorithm>.py -g \
        <generations> -p <population per core> <modelfile>"

    # read the given parameters
    try:
        options, files = getopt.getopt(
                sys.argv[1:],
                "hg:p:",
                ["help", "generations=", "population="])
    except getopt.GetoptError:
        print(usage_string)
        sys.exit(1)

    gen = 10
    pop = 100
    f_model = './JSPEval/xml/example.xml'
    for opt, arg in options:
        if opt in ("-h", "--help"):
            print(usage_string)
            sys.exit()
        elif opt in ('-g', '--generations'):
            gen = int(arg)
        elif opt in ('-p', '--population'):
            pop = int(arg)

    if len(files) > 0:
        f_model = files[0]

    print('Algorithm using:\ngenerations: {}\
\npopulation: {}\nmodelfile: {}'.format(gen, pop, f_model))

    return (gen, pop, f_model)
