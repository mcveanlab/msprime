#
# Copyright (C) 2014 Jerome Kelleher <jerome.kelleher@well.ox.ac.uk>
#
# This file is part of msprime.
#
# msprime is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# msprime is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with msprime.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Command line interfaces to the msprime library.
"""
from __future__ import division
from __future__ import print_function

import sys
import struct
import random
import argparse

mscompat_description = """\
An ms-compatible interface to the msprime library. Supports a
subset of the functionality available in ms.
"""

def get_seeds(random_seeds):
    """
    Takes the specified command line seeds and truncates them to 16
    bits if necessary. Then convert them to a single value that can
    be used to seed the python random number generator, and return
    both values.
    """
    max_seed = 2**16 - 1
    if random_seeds is None:
        seeds = [random.randint(1, max_seed) for j in range(3)]
    else:
        # Follow ms behaviour and truncate back to shorts
        seeds = [s if s < max_seed else max_seed for s in random_seeds]
    # Combine the together to get a 64 bit number
    seed = struct.unpack(">Q", struct.pack(">HHHH", 0, *seeds))[0]
    return seed, seeds

def run_simulations(args):
    """
    Runs the simulations according to the specified arguments.
    """
    # The first line of ms's output is the command line.
    print(" ".join(sys.argv))
    python_seed, ms_seeds = get_seeds(args.random_seeds)
    print(python_seed)
    random.seed(python_seed)
    print(" ".join(str(s) for s in ms_seeds))
    for j in range(args.num_replicates):
        print()
        print("\\\\")

def positive_int(value):
    int_value = int(value)
    if int_value <= 0:
        msg = "{0} in an invalid postive integer value".format(value)
        raise argparse.ArgumentTypeError(msg)
    return int_value

def mscompat_main():
    parser = argparse.ArgumentParser(description=mscompat_description)
    parser.add_argument("sample_size", type=positive_int, help="Sample size")
    parser.add_argument("num_replicates", type=positive_int,
            help="Number of independent replicates")

    group = parser.add_argument_group("Behaviour")
    group.add_argument("--mutation-rate", "-t", type=float, metavar="theta",
            help="Mutation rate theta=4*N0*mu")
    group.add_argument("--trees", "-T", action="store_true",
            help="Print out trees in Newick format")

    group = parser.add_argument_group("Demography")
    group.add_argument("--growth-rate", "-G", metavar="alpha", type=float,
            help="Population growth rate alpha.")
    group.add_argument("--growth-event", "-eG", nargs=2,
            metavar=("t", "alpha"),
            help="Set the growth rate to alpha at time t")
    group.add_argument("--size-event", "-eN", nargs=2,
            metavar=("t", "x"),
            help="Set the population size to x * N0 at time t")
    group = parser.add_argument_group("Miscellaneous")
    group.add_argument("--random-seeds", "-seeds", nargs=3, type=positive_int,
            metavar=("x1", "x2", "x3"),
            help="Random seeds (must be three integers)")
    group.add_argument("--precision", "-p", type=positive_int,
            help="Number of values after decimal place to print")
    args = parser.parse_args()
    if args.mutation_rate is None and not args.trees:
        parser.error("Need to specify at least one of --theta or --trees")

    run_simulations(args)
