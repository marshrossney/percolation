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
parser.add(
    "-n",
    "--n-days",
    type=int,
    required=True,
    help="number of 'days' (time steps) to evolve the model forwards",
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
    "--directed",
    type=str,
    choices=("isotropic", "right", "down", "both"),
    default="isotropic",
    help="Directed connections along none, one, or both of the axes, default: 'isotropic'",
)
parser.add(
    "--periodic",
    action="store_true",
    help="periodic boundary conditions on the lattice",
)
parser.add(
    "--critical-threshold",
    type=float,
    default=0.1,
    help="threshold for the fraction of infected nodes, used in plot and diagnostics, default: 0.1",
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

args = parser.parse_args()
