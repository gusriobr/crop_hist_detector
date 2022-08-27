import logging

import pandas as pd
import numpy as np

from cropseq import cfg
from cropseq.data.transform_codes import read_symbology_file

cfg.configLog()

"""
Reads the sample points and the mapping file and creates the final sequence dataset.
Input files:
- resources/code_mapping.xlsx: datatame with one column per year that maps the codes of that eyar with the reference year (last = 2021)
- resources/samples_sequence.pickle: panda dataframe with one row per each sample point (aprox 2M points) and one column per each 
year with the code of the detected land usage in that point for the year of the column.
Output file:
- resources/dataset.pickle: pandas dataframe with the points from the samples_sequence.pickle with the codes of each year
converted to reference year land usage codes.    
"""


def get_column_mapping(df_mapping, year):
    """
    Return a map that gives for each current year code the corresponding value in the last year symbology file
    ej. code in 2011 --> code in 2021, code in 2012 --> code in 2021, ...
    :param df_mapping:
    :param year:
    :return:
    """
    column_name = "code_{}".format(year)
    df_m = df_mapping[["code", column_name]].drop_duplicates()
    mapping = {}
    for idx, row in df_m.iterrows():
        if not pd.isna(row[column_name]):
            mapping[int(row[column_name])] = int(row["code"])
    return mapping


def run_transform():
    # read code mappings
    mapping_file = cfg.resource("code_mapping.xlsx")
    df_mapping = pd.read_excel(mapping_file)
    # open sequence file and transform each column
    seq_file = cfg.resource("samples_sequence.pickle")
    df_sequence = pd.read_pickle(seq_file)
    for year in range(2011, 2021):
        col_mapping = get_column_mapping(df_mapping, year)
        logging.info("Transforming codes for year {}".format(year))
        df_sequence[str(year)] = df_sequence[str(year)].apply(lambda x: col_mapping.get(x, x))
        df_sequence[str(year)] = df_sequence[str(year)].astype(np.uint8)

    # read year reference symbology file
    ref_symb_file = cfg.resource("data/codes/codes_2021.csv")
    df_symb = read_symbology_file(ref_symb_file, cols=['code', 'cubierta', 'land_usage'])
    df_symb["code"] = df_symb["code"].astype(np.uint8)

    # add names to the sequence
    df_sequence = df_sequence.merge(df_symb, how="left", left_on="2021", right_on="code")
    df_sequence = df_sequence.drop(["code", "2022"], axis=1)
    df_sequence.to_pickle(cfg.resource("dataset.pickle"))

    logging.info("Dataset created successfully!")

if __name__ == '__main__':
    run_transform()
