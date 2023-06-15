library(tidyverse)
library(aslib)

## overview of ASLib-related weblinks: https://www.coseal.net/aslib/

## import 
# as = aslib::getCosealASScenario("ASP-POTASSCO")
as = aslib::parseASScenario("data/TSP-ECJ2018")

## extract information of AS scenarios
algo = as$algo.runs
feat = as$feature.values

## visualizations
aslib::plotAlgoPerfScatterMatrix(as)
aslib::plotAlgoPerfCDFs(as)
aslib::plotAlgoPerfDensities(as) + scale_x_log10()
aslib::summarizeFeatureValues(as, type = "instance")
aslib::plotAlgoCorMatrix(as)
