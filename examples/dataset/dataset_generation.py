"""
This file demonstrates how to generate a benchmarking dataset from a path
pointing to the location of the test set using the DatasetGenerator class.

WARNING: Before running the dataset generator, please ensure that your system
has enough storage for the benchmarking dataset. The size of the dataset will
be roughly T x S x X where T is the number of unique transforms used, S is the
number of severity levels, and X is the original size of the test set. These
datasets can be quite large, especially for volumetric data.

Please refer to input_file_globbing.py for more detailed examples of how to
generate a list of input files to supply to the dataset generator.
"""

from pathlib import Path

from roodmri.data import DatasetGenerator

# get file list (see .input_file_globbing.py for more detailed examples)
data_dir = Path('/home/user/data/')
image_paths = [str(path) for path in sorted(data_dir.glob('test_data/*/t1.nii.gz'))]
label_paths = [str(path) for path in sorted(data_dir.glob('test_data/*/seg_label.nii.gz'))]
input_files = [{'image': img, 'label': lbl} for img, lbl in zip(image_paths, label_paths)]

# specify output path for benchmarking datsaset
out_path = '/home/user/data/benchmarking'

# initialize dataset generator
generator = DatasetGenerator(input_files, out_path)

# run generator and save filename mappings
generator.generate_dataset()
generator.save_filename_mappings(Path(out_path) / 'filename_mappings.csv')
