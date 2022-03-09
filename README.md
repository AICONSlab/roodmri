# **ROOD-MRI**: **B**enchmarking the **R**obustness of deep learning segmentation models to **O**ut-**O**f-**D**istribution data in **MRI**

![image](https://user-images.githubusercontent.com/22750822/157528340-d112408c-e3de-4ce7-87f8-8afb57d8d3da.png)

# Getting started

The [examples](https://github.com/AICONSlab/roodmri/tree/main/examples) folder contains examples of working with the `DatasetGenerator` class and calculating metrics from a dataframe of segmentation results using the `calculate_metrics` function.

The basic workflow is:

1. Generate a benchmarking dataset from a pre-existing test set using the `DatasetGenerator` class
2. Evaluate a trained model on the benchmarking dataset, generating a .csv
or dataframe containing segmentation metrics for each sample
3. Calculate benchmarking/robustness metrics from the .csv/dataframe generated
in step 2 using `calculate_metrics`

## 1. Generate a benchmarking dataset

Skip this step if you're using a pre-existing benchmarking dataset...

If you have a dataset directory that looks like this:

```
/home/user/data/
|-- train_data
`-- test_data
    |-- case_01
    |   |-- t1.nii.gz
    |   `-- seg_label.nii.gz
    |-- case_02
    |   |-- t1.nii.gz
    |   `-- seg_label.nii.gz
    .
    .
    .
    `-- case_99
        |-- t1.nii.gz
        `-- seg_label.nii.gz
```

First, glob the files into a structured list:

```
from pathlib import Path

data_dir = Path('/home/user/data/')
image_paths = [str(path) for path in sorted(data_dir.glob('test_data/*/t1.nii.gz'))]
label_paths = [str(path) for path in sorted(data_dir.glob('test_data/*/seg_label.nii.gz'))]
input_files = [{'image': img, 'label': lbl} for img, lbl in zip(image_paths, label_paths)]
```

Then, run the `DatasetGenerator` over the input files:

```
from roodmri.data import DatasetGenerator

out_path = '/home/user/data/benchmarking'   # specify the path to put benchmarking samples

generator = DatasetGenerator(input_files, out_path)
generator.generate_dataset()
generator.save_filename_mappings(Path(out_path) / 'filename_mappings.csv')   # save new filename mappings
```

The folder specified by `out_path` will now be populated with sub-folders named `Affine_1`, `Affine_2`, ..., `RicianNoise_4`, `RicianNoise_5`, ... containing transformed samples from the test set. In the name `RicianNoise_4`, `RicianNoise` refers to the transform applied and `4` refers to the severity level.

For more details and examples using different initial directory structures, see the [examples/dataset](https://github.com/AICONSlab/roodmri/tree/main/examples/dataset) folder.

## 2. Evaluate your model(s) on the benchmarking dataset

The end result of this step should be a csv file or dataframe with segmentation results for each benchmarking sample, as well as the original clean test set:

```
Model,Task,Transform,Severity,Subject_ID,DSC,HD95
unet_a,WMHs,Affine,1,Subject_001,0.82,1.41
unet_a,WMHs,Affine,1,Subject_002,0.79,2.34
.
.
unet_a,WMHs,Clean,0,Subject_001,0.85,1.41
.
.
unet_f,Ventricles,Affine,1,Subject_001,0.90,1.56
.
.
```

Since users' own evaluation pipelines may vary significantly (pre-processing, transforms, dataloaders, etc.), we do not provide modules to evaluate models on the benchmarking dataset. Rather, we suggest that users use their own existing pipelines to generate a csv file such as the one above. We will be uploading some of our own examples to the [examples](https://github.com/AICONSlab/roodmri/tree/main/examples) folder, including code for how to parse the transform/severity level folder name.

For more details regarding the requirements for the csv/dataframe, see [metric_calculations.py](https://github.com/AICONSlab/roodmri/blob/main/examples/metrics/metric_calculations.py) in the [examples](https://github.com/AICONSlab/roodmri/tree/main/examples) folder.

## 3. Calculate benchmarking metrics

After producing a csv/dataframe with segmentation results, you can use the `calculate_metrics` function to generate a suite of benchmarking metrics:

```
from pathlib import Path

import pandas as pd

from roodmri.metrics import calculate_metrics

data_path = '/home/user/data/model_evaluation_results.csv'   # change to location of csv
save_path = '/home/user/benchmarking/'   # change to desired location of output files
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
```

For more documentation, see [metric_calculations.py](https://github.com/AICONSlab/roodmri/blob/main/examples/metrics/metric_calculations.py) in the [examples](https://github.com/AICONSlab/roodmri/tree/main/examples) folder, or [calculate.py](https://github.com/AICONSlab/roodmri/blob/main/roodmri/metrics/calculate.py) which contains the `calculate_metrics` function. For metric formulations and how to use them, check out our paper.

