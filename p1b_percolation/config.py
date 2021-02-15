import configargparse

parser = configargparse.ArgParser()

parser.add("-c", "--config", is_config_file=True, help="path to config file")

parser.add(
    "-d",
    "--dimensions",
    type=int,
    action="append",
    required=True,
    help="number of nodes along each axis of the square lattice",
)
parser.add(
    "-o",
    "--outpath",
    type=str,
    default=".",
    help="path to directory for output files, default: '.'",
)
parser.add(
    "-n",
    "--n-steps",
    type=int,
    default=100,
    help="number of steps to evolve the model forwards",
)
parser.add(
    "-t",
    "--time",
    action="store_true",
    help="report the time taken to evolve the model",
)
parser.add(
    "-f",
    "--frozen-prob",
    type=float,
    default=0.0,
    help="probability for nodes to be initially flagged as frozen against the virus, default: 0",
)

parser.add(
    "--transmission-prob",
    type=float,
    default=1.0,
    help="probability of a live node transmitting to a susceptible contact upon a single refresh of the model, default: 1.0",
)
parser.add(
    "--recovery-time",
    type=int,
    default=1,
    help="number of time steps before a live node is considered to have recovered, and is no longer able to transmit",
)

parser.add(
    "--recovered-are-frozen",
    action="store_true",
    help="nodes which have recovered are flagged as frozen",
)
parser.add(
    "--shuffle-prob",
    type=float,
    default=0.0,
    help="probability for any given node to to shuffle positions with all other shuffling nodes at any given time, default: 0.0",
)
parser.add(
    "--nucleus-size",
    type=int,
    default=1,
    help="linear size of the initial live nucleus",
)
parser.add(
    "--n-links",
    type=int,
    choices=(1, 2, 3, 4),
    default=1,
    help="Number of links for each node on the lattice, default: 1",
)
parser.add(
    "--periodic",
    action="store_false",
    help="periodic boundary conditions on the lattice",
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

parser.add(
    "--parameter",
    type=str,
    default="frozen_prob",
    help="Parameter to scan over, default: 'frozen_prob'",
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
