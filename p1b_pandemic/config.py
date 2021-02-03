import configargparse
from sys import maxsize

parser = configargparse.ArgParser()

parser.add("-c", "--config", is_config_file=True, help="path to config file")

# Required args
parser.add(
    "-l",
    "--length",
    type=int,
    required=True,
    help="Number of nodes along one axes of the square lattice.",
)
parser.add(
    "-n",
    "--n-days",
    type=int,
    required=True,
    help="Number of 'days' (time steps) to evolve the model forwards.",
)

# Optional args
parser.add(
    "-o",
    "--outpath",
    type=str,
    default=".",
    help="Path to directory for output files. Default: '.'.",
)

parser.add(
    "-t",
    "--time",
    action="store_true",
    help="Report the time taken to evolve the model. Also forces --skip-animation=True.",
)
parser.add(
    "-p",
    "--transmission-prob",
    type=float,
    default=1.0,
    help="Probability of an infected node transmitting the virus to a susceptible contact upon a single refresh of the model. Default: 1.0.",
)
parser.add(
    "-v",
    "--vaccine-frac",
    type=float,
    default=0.0,
    help="Fraction of nodes that are initially flagged as immune against the virus. Default: 0.",
)
parser.add(
    "-r",
    "--recovery-time",
    type=int,
    default=maxsize,
    help="Number of time steps before an infected node is considered to have recovered, and is no longer able to spread the infection.",
)

parser.add(
    "--recovered-are-immune",
    action="store_true",
    help="Nodes which have recovered from the infection are flagged as immune.",
)
parser.add(
    "--initial-shape",
    type=str,
    choices=("nucleus", "line"),
    default="nucleus",
    help="'Shape' of the infection on day 0. Choices: 'nucleus' (a square), 'line'.",
)
parser.add(
    "--nucleus-size",
    type=int,
    default=1,
    help="Linear size of the initial infected nucleus, i.e. size length of the square.",
)
parser.add(
    "--periodic",
    action="store_true",
    help="Periodic boundary conditions on the lattice.",
)
parser.add(
    "--skip-animation",
    action="store_true",
    help="Do not create an animation of the pandemic evolution.",
)
parser.add(
    "--seed",
    type=int,
    default=None,
    help="Specify a seed for the random number generator. For reproducibility.",
)

args = parser.parse_args()
