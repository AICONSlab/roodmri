from curses import savetty
from pathlib import Path

import pandas as pd

from roodmri.metrics.calculate import calculate_metrics

"""
Benchmarking/robustness metric calculation example.

This script demonstrates a simple example of using the calculate_metrics
function to calculate benchmarking/robustness metrics from a csv file containing
segmentation evaluation results in reference to a benchmarking dataset. The
csv file is located at data_path and resulting dataframes will be saved as csv
files in save_path. In this example, the structure of the csv file might appear
like the one below:

Model,Task,Transform,Severity,Subject_ID,DSC,HD95
unet_a,WMHs,Affine,1,Subject_001,0.82,1.41
unet_a,WMHs,Affine,1,Subject_002,0.79,2.34
...
unet_a,WMHs,Clean,0,Subject_001,0.85,1.41
...
unet_f,Ventricles,Affine,1,Subject_001,0.90,1.56
...

A couple of important things to note:
- The csv file can contain evaluation results for multiple models/tasks.
  Calculations will be performed separately for each model/task combination,
  or any other columns specified in grouping_cols.
- Each unique set of values for the grouping columns must contain rows for
  clean data (one row for each subject in the test set). These are used for
  benchmarking metric calculations.
- Other metrics besides DSC and HD95 can be used, and analagous benchmarking
  metrics to those in Boone et al., 2022 will be calculated for those metrics.
  It must be specified in the metric_cols dictionary whether higher values
  or lower values are preferred for each segmentation metric (e.g., {'DSC':
  True, 'HD95': False}).
- Depending on the format of your input csv file, you may need to tweak the
  arguments supplied to pd.read_csv. See the pandas library documentation for
  more details.
"""

data_path = '/home/user/data/model_evaluation_results.csv'
save_path = '/home/user/benchmarking/'
df = pd.read_csv(data_path)
transform_level_metrics, aggregated_metrics = calculate_metrics(
    df=df,
    transform_col='Transform',
    severity_col='Severity',
    metric_cols={'DSC': True, 'HD95': False},
    clean_label='Clean',
    grouping_cols=['Model', 'Task']
)
transform_level_metrics.to_csv(Path(save_path) / 'transform_level_metrics.csv')
aggregated_metrics.to_csv(Path(save_path) / 'aggregated_metrics.csv')
