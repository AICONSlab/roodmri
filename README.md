# roodmri

The basic workflow is:

1. Generate a benchmarking dataset
2. Evaluate a trained model on the benchmarking dataset, generating a .csv
or dataframe containing segmentation metrics for each sample
3. Calculate benchmarking/robustness metrics from the .csv/dataframe generated
in step 2.
