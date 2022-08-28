import csv
import json
import logging
import os

import numpy as np
import pandas as pd

from cropseq import cfg
from cropseq.data.extract import year_from_filepath

cfg.configLog()

"""
The land usage rasters don't have a common code set, this script extracts the codes from the symbology
CSV's and creates a table that maps last year codes with previous ones.
Input files: data/codes/codes_{year}.csv containing the list of codes for the year raster.
Output files:
 -  code_mapping.xlsx: excel that maps for each year, the code map between the year raster and the reference 
    raster (currently 2021)
 - unused_codes.json: dict containing for each year the list of codes that couldn't been mapped.

"""


def read_codes(df, path):
    """
    Reads the file and tries to match eah value of the current file with reference values
    :param df: reference dataframe to append the column with codes from the yearly file
    :param path: symbology file with yearly raster codes
    :return:
    """
    df_year = read_symbology_file(path)

    # merge dataframes by name
    df_year["cubierta_y"] = df_year["cubierta_y"].str.upper()
    # translate names
    df_year["cubierta_y"] = df_year["cubierta_y"].apply(lambda x: fixed_mapping.get(x, x))
    df_total = df.merge(df_year, how="left", left_on="cubierta_up", right_on="cubierta_y")
    year = year_from_filepath(path)
    year_column = "code_" + year
    df_total = df_total.rename(columns={"code_x": "code", "code_y": year_column})
    df_total = df_total.drop("cubierta_y", axis=1)
    # now find what codes in the original file couldn't been mapped
    original_codes = df_year.code.unique()
    mask = np.isin(original_codes, df_total[year_column].values)
    unused_codes = original_codes[~mask].tolist()
    return df_total, unused_codes


def read_symbology_file(path, cols=['code', 'cubierta_y']):
    df_year = pd.DataFrame(columns=cols)
    # read the csv file into a dataframe
    with open(path, 'r', encoding="iso-8859-15") as file:
        reader = csv.reader(file, delimiter=";")
        for idx, row in enumerate(reader):
            if idx == 0:  # skip header
                continue
            content = {}
            for i, col_name in enumerate(cols):
                content[col_name] = row[i]
            df_row = pd.DataFrame.from_dict([content])
            df_year = pd.concat([df_year, df_row])
    return df_year


def read_ref_file(path):
    df = pd.DataFrame(columns=['code', 'cubierta', 'land_usage'])
    with open(path, 'r', encoding="iso-8859-15") as file:
        reader = csv.reader(file, delimiter=";")
        for idx, row in enumerate(reader):
            if idx == 0:  # skip header
                continue
            print(row)
            content = {"code": row[0], "cubierta": row[1], "land_usage": row[2] if len(row) > 6 else None}
            df = pd.concat([df, pd.DataFrame.from_records([content], columns=['code', 'cubierta', 'land_usage'])])

    # uppercase columna name for comparations
    df["cubierta_up"] = df["cubierta"].str.upper()
    return df


fixed_mapping = {
    "ARTIFICIAL": "ARTIFICIALES",
    "URBANO-VIALES": "ARTIFICIALES",
    "OTRAS LEGUMINOSAS GRANO": "OTRAS LEGUMINOSAS",
    "GUISANTES": "GUISANTE",
    "CÁRTAMO": "CARTAMO",
    "CASTAÑARES": "CASTAÑOS",
    "TRIGO REGADIO": "TRIGO",
    "CEBADA REGADIO": "CEBADA",
    "ALFALFA REGADIO": "ALFALFA",
    "GIRASOL REGADIO": "GIRASOL",
    "CENTENO REGADIO": "CENTENO",
    "AVENA REGADIO": "AVENA",
    "TRITICALE REGADIO": "OTROS CEREALES",
    "OTROS CEREALES REGADIO": "OTROS CEREALES",
    "PATATAS": "PATATA",
    "HORTICOLA": "HORTICOLAS",
    "BARBECHO": "ERIAL",
    "HUERTA CEBOLLA": "CEBOLLA",
    "HUERTA": "HORTICOLAS",
    "FRUTALES DE CASCARA": "FRUTALES CASCARA",
    "OTROS FRUTALES": "FRUTALES",
    "PERALES": "FRUTALES",
    "MANZANOS": "FRUTALES",
    "POPULUS SPP.": "CHOPOS",
    "FAGUS SYLVATICA": "FRONDOSAS CADUCIFOLIAS",
    "FRONDOSAS SIEMPRE VERDES": "FRONDOSAS PERENNIFOLIAS",
    "EUCALYPTUS CAMALDULENSIS": "FRONDOSAS PERENNIFOLIAS",
    "QUERCUS ILEX": "FRONDOSAS PERENNIFOLIAS",
    "QUERCUS FAGINEA": "FRONDOSAS PERENNIFOLIAS",
    "QUERCUS PYRENAICA": "FRONDOSAS PERENNIFOLIAS",
    "QUERCUS ROBUR": "FRONDOSAS PERENNIFOLIAS",
    "PINUS SYLVESTRIS": "CONIFERAS",
    "PINUS NIGRA": "CONIFERAS",
    "PINUS PINASTER": "CONIFERAS",
    "PINUS PINEA": "CONIFERAS",
    "PINUS HALEPENSIS": "CONIFERAS",
    "PINUS RADIATA": "CONIFERAS",
    "JUNIPERUS THURIFERA": "CONIFERAS",
    "TRITICALE": "OTROS CEREALES"
}

def manual_code_revision(df):
    """
    Some codes used in year 2021 aren't abailable in previous years, it seems, some categories have been extracted from bigger groups like lentis from leguminous.
    This new categories are available just in the last two years. To keep the categories homogeneous as much as possible, this land usages are merged back to the main groups.
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

# read last year codes
if __name__ == '__main__':
    codes_folder = cfg.resource('data/codes')
    csv_files = [os.path.join(codes_folder, f) for f in os.listdir(codes_folder)]
    csv_files.sort()
    csv_files.reverse()
    # get last year CSV file and take its codes as reference to compare with previous files
    df = read_ref_file(csv_files[0])
    csv_files = csv_files[1:]

    unused_codes = {}
    for file in csv_files:
        df, unused = read_codes(df, file)
        year = year_from_filepath(file)
        unused_codes[year] = unused

    output_file = cfg.resource("code_mapping.xlsx")
    df.to_excel(output_file)

    # store as json the codes that we couldn't map
    with open(cfg.resource("unused_codes.json"), 'w') as fout:
        json_dumps_str = json.dumps(unused_codes, indent=4)
        print(json_dumps_str, file=fout)

    logging.info("Successfully created land usage codes map, output file: {}".format(output_file))
