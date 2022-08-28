Land usage detection from historical data
--------------------------------------------

This is a pet project to apply machine learning and bayesian techniques to predict the next year land usage based on the
last 8-10 year historical data. The project is focused on the area of Castilla y León, a Spanish region ....

# Data sources

This project uses the information publicly provided by the Instituto Tecnológico Agrario de Castilla y
León (www.itacyl.es) through the website of the project Mapa de Cultivos y Superficies Naturales de Castilla y León (
MCSNCyL) (https://mcsncyl.itacyl.es/). The **Map of Crops and Natural Surfaces of Castilla y León (MCSNCyL)** is a map
of land cover, obtained through satellite images and updated annually with two biannual versions. The objective is to
have a mapping of land use that represents the changes in annual herbaceous crops, the majority in the Autonomous
Community, and in natural vegetation surfaces. The project began in 2013, and since then maps have been generated from
2011 to the present. In addition, since 2020 two different layers have been published, one with the type of crop or area
identified and the other with the exploitation system (rainfed/irrigated) identified by remote sensing. For the
elaboration of the map, images from the Deimos-1 (2011-2016), Landsat 8 (2013-2016), Sentinel-2 (2016-present)
Satellites have been used. As of 2017, Sentinel-2A and Sentinel-2B images have been used, which entails an improvement
in the spatial resolution of the product from 20m to 10m.

# Preparing the environment

To extract raster information you first have to download the raster files from the ITACyL ftp
in https://ftp.itacyl.es/Atlas_Agroclimatico/03_ActividadAgraria/Agricultura/CultivosYSuperfNaturales. Download all the
${year}_Clasificacion_MCSNCyL.zip file to store the .tiff files in the RESOURCES/DATA folder To run this code, the
MCSNCyL raster are expected to be in the folder cfg.RASTER_FOLDER

In this project the raster coberturas of the last 10 years have been downloaded to extract the predicted crop code

# Dataset

The input dataset for this project is a sequence of land usage codes for the las 10 years (2011-2021). The dataset has
been obtained following these steps:

- First 1 million points have been chosen randomly in the area of Castilla y León.
- For each point, the coordinates are used to extract the pixel value (land usage code) from each yearly raster,
  obtaining a 10-year code sequence.
- Once the sequence is obtained, the code are reviewed and mapped to current year code list (2021) using the raster
  symbology information (relation between code and land usage). This way each land usage is referred using the same code
  among all the years.

The land usage codes is not homogeneous among all the years, the script `transform_codes.py` includes the code to map
each year code list to the reference year code table. (Currently 2021).

Final dataset is found in [resources/dataset.pickle](resources/dataset.pickle), is a panda dataframe containing more
that 2M points. To create the dataset run these scripts in this order:

1. `extract.py`: extract the sequencce points from shp and get codes for earch year raster.
2. `transform_codes.py`: transform all codes to current year reference code table.
3. `create_dataset.py`: create final dataset with code conversion file created in step 2.

A small exploratory analysis is made in [dataset_review.ipynb](src/cropseq/data/notebooks/dataset_review.ipynb).

# Analyzing the data

Conclusions