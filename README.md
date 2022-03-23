# **ROOD-MRI**: **B**enchmarking the **R**obustness of deep learning segmentation models to **O**ut-**O**f-**D**istribution data in **MRI**

![image](https://user-images.githubusercontent.com/22750822/157528340-d112408c-e3de-4ce7-87f8-8afb57d8d3da.png)

# Getting started

The [examples](https://github.com/AICONSlab/roodmri/tree/main/examples) folder contains example scripts for working with the `DatasetGenerator` class and `calculate_metrics` function. We will be adding more examples to this folder, stay tuned!

The basic workflow is:

1. Generate a benchmarking dataset from a pre-existing test set using the `DatasetGenerator` class.
2. Evaluate a trained model on the benchmarking dataset, generating a
dataframe or .csv file containing segmentation metrics for each sample.
3. Calculate benchmarking/robustness metrics from the dataframe/.csv generated
in step 2 using `calculate_metrics`.

Dependencies for this library are listed in the [requirements.txt](https://github.com/AICONSlab/roodmri/blob/main/requirements.txt) file (including [MONAI](https://github.com/Project-MONAI/MONAI) and [TorchIO](https://github.com/fepegar/torchio)).

## 1. Generate a benchmarking dataset

Skip this step if you're using a pre-existing benchmarking dataset (see links to existing datasets below).

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

The folder specified by `out_path` will now be populated with sub-folders named `Affine_1`, `Affine_2`, ..., `RicianNoise_4`, `RicianNoise_5`, ... containing transformed samples from the test set. In the name `RicianNoise_4`, `RicianNoise` refers to the transform applied and `4` refers to the severity level. The image below illustrates an example of the five default severity levels on a sample T1-weighted image for (a) ghosting, (b) isotropic downsampling, and (c) MRI (Rician) noise:

![image](https://user-images.githubusercontent.com/22750822/157904816-5c2ea501-9059-4783-a590-d61c947fda82.png)

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

The image below demonstrates an example of using benchmarking metrics to comparing model architectures. The numbers in the lower- and upper-left corners of the top-row and bottom-row subplots, respectively, correspond to the mean degradation for each model (top row: Dice similarity coefficient; bottom row: modified (95th percentile) Hausdorff distance):

![image](https://user-images.githubusercontent.com/22750822/157905587-e475c048-d2a7-453c-8d8d-8020afba9e6c.png)

For more documentation, see [metric_calculations.py](https://github.com/AICONSlab/roodmri/blob/main/examples/metrics/metric_calculations.py) in the [examples](https://github.com/AICONSlab/roodmri/tree/main/examples) folder, or [calculate.py](https://github.com/AICONSlab/roodmri/blob/main/roodmri/metrics/calculate.py) which contains the `calculate_metrics` function. For metric formulations and how to use them, check out our paper.

# Links to existing datasets

See the list below for download links to existing benchmarking datasets:

- Hippocampus segmentation dataset: https://www.dropbox.com/sh/t0id61jfwdq1dp9/AAAJyQLUP_6RSFjp-UOfa-Lxa?dl=0

# Credits

**If you like this repository, please click on Star!**

If you use this package for your research, please cite our paper:

[Boone, L., Biparva, M., Forooshani, P. M., Ramirez, J., Masellis, M., Bartha, R., ... & Goubran, M. (2022). ROOD-MRI: Benchmarking the robustness of deep learning segmentation models to out-of-distribution and corrupted data in MRI. arXiv preprint arXiv:2203.06060.](https://arxiv.org/abs/2203.06060)

```
@article{boone2022rood,
  title={ROOD-MRI: Benchmarking the robustness of deep learning segmentation models to out-of-distribution and corrupted data in MRI},
  author={Boone, Lyndon and Biparva, Mahdi and Forooshani, Parisa Mojiri and Ramirez, Joel and Masellis, Mario and Bartha, Robert and Symons, Sean and Strother, Stephen and Black, Sandra E and Heyn, Chris and others},
  journal={arXiv preprint arXiv:2203.06060},
  year={2022}
}
```

This work has been enabled by the flexibility and modularity of the [MONAI](https://github.com/Project-MONAI/MONAI) and [TorchIO](https://github.com/fepegar/torchio) libraries. If you like our work and aren't familiar with theirs already, go check them out!
