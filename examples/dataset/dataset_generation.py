# Copyright (C) 2022  AICONS Lab

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
