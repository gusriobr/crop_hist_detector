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

# Analyzing the data

The exploratory analysis is made in [dataset_review.ipynb](src/cropseq/data/notebooks/dataset_review.ipynb). Whe can
summarize the findinds as:

- First, we have to take in count that the ground-truth use in this project is the output of the classifier, so the
  dataset will include a variance source derived from error in the classifier prediction, and there's no way to measure
  this epistemic error.
- Percentage 36.71% of sample points have a code that can be grouped as Not cultivated área: ("Bare soil", "Scrub", "
  Grassland", "Artificial", etc.).
- Some crops are overrepresented and gather around 80% of the cultivated area: wheat, barley, suflower, maize and oat
  mainly.
- About the land usage variation among years, as mean, 65% of the records show a change in the land usage.
- If we put apart the land usages that are not expected to change (ex: woody crops, trees, water bodies, vineyard,
  olive, rocky areas, etc), this percentage raises to 70%.
- Analyzing the possible combinations included in the dataset, the top combinations add up 53% of total variations
  represented in the samples.
- With this information, we can build a conditional distribution that rules the passing from one land usage to other as
  the conditional probability: p(land_usage<sub>i</sub> | land_usage<sub>j</sub>). We can calculate these probabilities
  just counting the variations from one land usage to others, for example we know that **25% of the times a farmer grows
  wheat, next year he/she grows barley**. These probabilities can be used to create the base model.

# Model 0: base model

Using the land usage variations found in the dataset we can create a table that gives us the probability of each crop on
the next year. Using this table as conditional probability we can simulate the prediction of the next year land usages.

We create the CDF with data from years 2011-2020 and predict on 2021. As expected, the model performs poorly with a
**f1-score=0.29**. The model can predict acceptably the stable land usage categories (trees, artificial surfaces,
vineyard, etc.) and over-represented crops (wheat and barley). As we have seen, the variation of crops is around 70% so
prediction based just on last year is not an acceptable option.

[Base model](src/cropseq/data/notebooks/dataset_review.ipynb#Estimating-base-model)

# Model 1: Hidden Markov Models

In this case HMM let us calculate the conditional probability taken into account he full sequence.

Transition matrix In addition to using the forward-backward algorithm to just calculate posterior probabilities for each
observation, we can count the number of transitions that are predicted to occur between the hidden states.

This is the transition table, which has the soft count of the number of transitions across an edge in the model given a
single sequence. It is a square matrix of size equal to the number of states (including start and end state), with
number of transitions from (row_id) to (column_id).

**After running the experiment for multiple possible states, and for each state multiple times to avoid getting stuck in
local minima, the best number of state seems to be 12. The log-prob keeps increasing with higher number of states, but
the classifier metrics (f1 and kappa) measured on the test split start going downwards what might be and indicator for
model over-fitting. See  [evaluate_model.ipynb](src/cropseq/hmm/notebooks/evaluate_model.ipynb) for detailed data about
the model fitting and state selection.

Final model transition can be visualized using graphviz as a svg
file: [Final model graph](resources/docs/hmm/final_model_plot.svg)

Having set the number of hidden states, next step is to train the model multiple times with full data and measure the
prediction performance of the model. After this process the final model outcomes **f1-score=0.43 and kappa= 0.38**. The
model outperforms the base model but keeps performing poorly.

# References

https://github.com/jmschrei/pomegranate/blob/master/tutorials/B_Model_Tutorial_3_Hidden_Markov_Models.ipynb

https://stats.stackexchange.com/questions/71940/hidden-markov-model-to-predict-the-next-state

Pomegranate
https://medium.com/analytics-vidhya/how-to-build-a-simple-hidden-markov-models-with-pomegranate-dfa885b337fb
https://notebook.community/jmschrei/pomegranate/tutorials/B_Model_Tutorial_3_Hidden_Markov_Models

De Souza e Silva, Edmundo & Leão, Rosa & Muntz, Richard. (2010). Performance Evaluation with Hidden Markov Models.
112-128. 10.1007/978-3-642-25575-5_10.
https://www.researchgate.net/publication/221152778_Performance_Evaluation_with_Hidden_Markov_Models
