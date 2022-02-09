# roodmri

roodmri is a platform for benchmarking the **R**obustness of deep learning segmentation models to **O**ut-**O**f-**D**istribution data in **MRI**.

The basic workflow is:

1. Generate a benchmarking dataset from a pre-existing test set using the `DatasetGenerator` class
2. Evaluate a trained model on the benchmarking dataset, generating a .csv
or dataframe containing segmentation metrics for each sample
3. Calculate robustness metrics from the .csv/dataframe generated
in step 2 using `calculate_metrics`

## Getting started

The [examples](https://github.com/AICONSlab/roodmri/tree/main/examples) folder contains examples of working with the `DatasetGenerator` class and calculating metrics from a dataframe of segmentation results using the `calculate_metrics` function.
