import re
import argparse
from io import StringIO
from itertools import accumulate

import pandas as pd

def dmp_to_df(dmp: str):
    names = re.sub(
        r"\t\|$", "", dmp, flags=re.MULTILINE
    )  # remove \t| from the end of each line
    names = re.sub(r"(\t)?\|(\t)?", "|", names)  # strip tabs surrounding | delimiters
    return pd.read_csv(StringIO(names), sep="|", header=None)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--names', help='The file containing species name data.', required=True)
    parser.add_argument('--nodes', help='The file containing node data.', required=True)

    args = parser.parse_args()

    with open(args.names) as f:
        names_df = dmp_to_df(f.read())

    with open(args.nodes) as f:
        nodes_df = dmp_to_df(f.read())

    names_df = names_df[
        [0, 1, 3]
       ]  # select columns containing node id, name, and name type (scientific, common, etc.), respectively

    nodes_df = nodes_df[
        [0, 2]
       ]  # select columns containing node id and node rank (species, genus, etc.) respectively

    names_and_nodes_df = names_df.merge(nodes_df, how="left", on=[0])

    names_and_nodes_species_df = names_and_nodes_df[
        names_and_nodes_df[3].eq("scientific name") & names_and_nodes_df[2].eq("species")
       ]

    species_names_df = names_and_nodes_species_df[1].reset_index(drop=True)

    species_names_df = species_names_df[
        species_names_df.str.fullmatch("^[A-Za-z]+ [A-Za-z]+$")
       ]  # filter out species that don't match the pattern of two words with only letters, separated by a space

    species_names_df = species_names_df[
        species_names_df.str.len().le(35)
       ]  # filter out really long names. sorry, Myxococcus llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogochensis

    #print(species_names_df)
    species_names_df.to_csv("data/species.csv", header=False, index=False)


if __name__ == '__main__':
    main()
