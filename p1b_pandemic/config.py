import configargparse
from sys import maxsize

parser = configargparse.ArgParser()

parser.add("-c", "--config", is_config_file=True, help="path to config file")

# Required args
parser.add(
    "-d",
    "--dimensions",
    type=int,
    action="append",
    required=True,
    help="number of nodes along each axis of the lattice",
)
# Optional args
parser.add(
    "-o",
    "--outpath",
    type=str,
    default=".",
    help="path to directory for output files, default: '.'",
)
parser.add(
    "-n",
    "--n-days",
    type=int,
    default=100,
    help="number of 'days' (time steps) to evolve the model forwards",
)
parser.add(
    "-t",
    "--time",
    action="store_true",
    help="report the time taken to evolve the model (forces --skip-animation=true)",
)
parser.add(
    "-p",
    "--transmission-prob",
    type=float,
    default=1.0,
    help="probability of an infected node transmitting the virus to a susceptible contact upon a single refresh of the model, default: 1.0",
)
parser.add(
    "-v",
    "--vaccine-frac",
    type=float,
    default=0.0,
    help="fraction of nodes that are initially flagged as immune against the virus, default: 0",
)
parser.add(
    "-r",
    "--recovery-time",
    type=int,
    default=maxsize,
    help="number of time steps before an infected node is considered to have recovered, and is no longer able to spread the infection",
)

parser.add(
    "--recovered-are-immune",
    action="store_true",
    help="nodes which have recovered from the infection are flagged as immune",
)
parser.add(
    "--travel-prob",
    type=float,
    default=0.0,
    help="probability for any given node to 'travel', which is to shuffle positions with all other travelling nodes at any given time, default: 0.0",
)
parser.add(
    "--initial-shape",
    type=str,
    choices=("nucleus", "line"),
    default="nucleus",
    help="'shape' of the infection on day 0, choices: 'nucleus', 'line'",
)
parser.add(
    "--nucleus-size",
    type=int,
    default=1,
    help="linear size of the initial infected nucleus",
)
parser.add(
    "--n-connections",
    type=int,
    choices=(1, 2, 3, 4),
    default=4,
    help="Number of connections for each node on the lattice, default: 4",
)
parser.add(
    "--periodic",
    action="store_false",
    help="periodic boundary conditions on the lattice",
)
parser.add(
    "--skip-animation",
    action="store_true",
    help="do not create an animation of the pandemic evolution",
)
parser.add(
    "--interval",
    type=int,
    default=25,
    help="number of milliseconds delay between each update in the animation",
)
parser.add(
    "--reproducible",
    action="store_true",
    help="If true, use a known seed for the random number generator",
)

# Parameter scan only
parser.add(
    "--parameter",
    type=str,
    default="vaccine_frac",
    help="Parameter to scan over, default: 'vaccine_frac'",
)
parser.add(
    "--start",
    type=float,
    default=0,
    help="Lower limit of the parameter to scan over, default: 0",
)
parser.add(
    "--stop",
    type=float,
    default=0.8,
    help="Upper limit of the parameter to scan over, default: 0",
)
parser.add(
    "--num",
    type=int,
    default=20,
    help="Number of values of the parameter to scan over, default: 20",
)
parser.add(
    "--repeats",
    type=int,
    default=10,
    help="Number of simulations to run for a given set of parameters, default: 10",
)
