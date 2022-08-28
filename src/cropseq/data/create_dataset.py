import logging

import numpy as np
import pandas as pd

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


def map_codes_to_current_year(df, df_mapping):
    for year in range(2011, 2021):
        col_mapping = get_column_mapping(df_mapping, year)
        logging.info("Transforming codes for year {}".format(year))
        # land usage codes without mapping in the current year will remain as 0
        df[str(year)] = df[str(year)].apply(lambda x: col_mapping.get(x, 0))
        df[str(year)] = df[str(year)].astype(np.uint8)
    return df


def manual_code_revision(df):
    """
    Some codes used in year 2021 aren't abailable in previous years, it seems, some categories have been extracted from
    bigger groups like lentis from leguminous. This new categories are available just in the last two years.
    To keep the categories homogeneous as much as possible, these land usages are merged back to the main groups.
    """
    merge_groups = {  # "GARBANZO": "OTRAS LEGUMINOSAS",
        54: 45,
        # "LENTEJAS": "OTRAS LEGUMINOSAS",
        55: 45,
        # ALUBIAS": "OTRAS LEGUMINOSAS",
        57: 45,
        # "YEROS": "FORRAJERAS",
        58: 11,
        # "ESPARCETA": "FORRAJERAS",
        67: 11,
        # "ZANAHORIA": "HORTICOLAS",
        60: 17,
        # "AJO": "HORTICOLAS",
        61: 17,
        # "CEBOLLA": "HORTICOLAS",
        62: 17,
        # "FRESAS": "HORTICOLAS",
        63: 17,
        # "PUERROS": "HORTICOLAS",
        64: 17,
    }
    # convertimos las descdripciones a códigos
    for year in range(2011, 2022):
        logging.info("Grouping codes into main land usage categories for year {}".format(year))
        year_col = str(year)
        df[year_col] = df[year_col].apply(lambda x: merge_groups.get(x, x))
        df[year_col] = df[year_col].astype(np.uint8)
    return df


def run_create_dataset():
    # read code mappings
    mapping_file = cfg.resource("code_mapping.xlsx")
    df_mapping = pd.read_excel(mapping_file)
    # open sequence file and transform each column
    seq_file = cfg.resource("samples_sequence.pickle")
    df_sequence = pd.read_pickle(seq_file)
    # filter samples that has a 0 in any column of the sequence, this means that the point has no value in the raster
    df_sequence = df_sequence[~df_sequence.isin([0]).any(axis=1)]

    df_sequence = map_codes_to_current_year(df_sequence, df_mapping)
    df_sequence = manual_code_revision(df_sequence)

    # read year reference symbology file
    ref_symb_file = cfg.resource("data/codes/codes_2021.csv")
    df_symb = read_symbology_file(ref_symb_file, cols=['code', 'cubierta', 'land_usage'])
    df_symb["code"] = df_symb["code"].astype(np.uint8)

    # remove data from 2022, symb file is missing
    df_sequence = df_sequence.merge(df_symb, how="left", left_on="2021", right_on="code")
    df_sequence = df_sequence.drop(["code", "2022"], axis=1)

    # add names to the sequence
    df_sequence.to_pickle(cfg.resource("dataset.pickle"))

    logging.info("Dataset created successfully!")

    return df_sequence


if __name__ == '__main__':
    df_dataset = run_create_dataset()
