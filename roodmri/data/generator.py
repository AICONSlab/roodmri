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

from copy import deepcopy
import csv
from pathlib import Path
import time

from monai.data import DataLoader, Dataset, write_nifti
import monai.transforms as tf
import numpy as np
import torch

from roodmri.data.utils import is_file_list
from roodmri.transforms.defaults import DEFAULT_TRANSFORM_SETTINGS
from roodmri.utils.misc import rgetattr, rsetattr

BASE_TRANSFORMS = [
    tf.LoadImaged(keys=['image', 'label']),
    tf.AddChanneld(keys=['image', 'label'])
]
DEVICE = torch.device("cpu")


class DatasetGenerator(object):
    """
    Class to generate a benchmarking datastet from a pre-existing test set.

    An object of the DatasetGenerator class has attributes and methods
    necessary to generate a benchmarking dataset. Attributes are described
    below. The main method is generate_dataset, which is called
    to iterate over a test set and save transformed/corrupted versions of each
    sample in the test set to a folder path specified in the attributes.

    Args:
        input_files: List of dictionaries containing the paths
            to images and labels in the test set. Each dictionary corresponds
            to a single test sample/patient case. Each dictionary should have
            two items with the following keys/values: 'image': full path to the
            image file (e.g., a brain MRI NIfTI file); 'label': full path to
            the 'ground truth' binary segmentation. For an example of how to
            produce a list like this, see input_file_globbing.py in the examples
            directory.
        out_path: string specifying the full path to the directory where the
            benchmarking dataset will be stored.
        transform_settings (optional): Dictionary containing transforms and settings for
            the severity levels to be used in generating the benchmarking
            dataset. For an example of the structure of such a dictionary, see
            the defaults in roodmri/transforms/defaults.py.

    Additional attributes:
        filename_mappings: Dictionary containing mappings from provided file
            names to newly created sample IDs in the benchmarking dataset. To
            maintain flexibility with different users' filename conventions,
            new subject IDs are generated in the benchmarking dataset in simple
            numerical order (e.g., '000001', '000002', ...). The
            filename_mappings object can be used to map back to original
            filename conventions for subsequent analyses.
    """

    def __init__(self, input_files, out_path, transform_settings = None):
        assert is_file_list(input_files)
        assert isinstance(out_path, str), "out_path should be a string."
        self.input_files = input_files
        self.out_path = Path(out_path)
        if not self.out_path.exists():
            self.out_path.mkdir(parents=True, exist_ok=True)
            print("WARNING: out_path does not already exist. A new directory "
                  "was created at the path specified by out_path. If permission"
                  " errors occur, please create a directory with the correct"
                  "permissions prior to initializing this object.")
        if transform_settings is not None:
            self.transform_settings = transform_settings
        else:
            self.transform_settings = DEFAULT_TRANSFORM_SETTINGS
        filename_mappings = dict()
        for i, sample_dict in enumerate(self.input_files):
            filename_mappings[sample_dict['label']] = str(i + 1).zfill(6)
        self.filename_mappings = filename_mappings
        print("WARNING: running the generate_dataset method may consume a lot"
              " of your machine's storage (roughly the size of the test set x "
              "the number of transforms x the number of severity levels). "
              "Please ensure that you have enough free space in the directory "
              "specified by out_path before continuing.")

    def generate_dataset(self):
        """
        Generate benchmarking samples.

        This method iterates over the test set specified by the file names in
        self.input_files, transforming the images for each sample according to
        transforms and severity levels specified in self.transform_settings.
        Files are saved in the folder specified by self.out_path.
        """
        for transform_name, settings in self.transform_settings.items():
            sv_controller = settings['severity_controller']
            # iterate over severity levels
            for i in range(len(sv_controller[next(iter(sv_controller))])):
                print("-" * 10)
                transform = deepcopy(settings['transform'])
                for param, values in sv_controller.items():
                    value = values[i]
                    rsetattr(transform, param, value)
                print(transform_name, *[(param, rgetattr(transform, param))
                                        for param in sv_controller.keys()])
                transforms = tf.Compose(BASE_TRANSFORMS + settings['pre_transforms']
                                        + [transform] + settings['post_transforms'])
                test_ds = Dataset(data=self.input_files, transform=transforms)
                loader = DataLoader(test_ds, batch_size=1)
                step_start = time.time()
                for j, test_data in enumerate(loader):
                    subject_id = self.filename_mappings[test_data['label_meta_dict']['filename_or_obj'][0]]
                    affine = test_data['image_meta_dict']['affine'][0]
                    save_dir = self.out_path / f'{transform_name}_{i+1}/{subject_id}'
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    test_inputs, test_labels = (
                        test_data["image"].to(DEVICE),
                        test_data["label"].to(DEVICE),
                    )
                    test_inputs = np.squeeze(test_inputs.numpy())
                    test_labels = np.squeeze(test_labels.numpy())
                    for data, label in zip([test_inputs, test_labels], ['image', 'label']):
                        write_nifti(
                            data,
                            save_dir / f'{subject_id}_{transform_name}_{i+1}_{label}.nii.gz',
                            affine=affine
                        )
                    step_time = time.time() - step_start
                    print(f"{j+1}/{len(test_ds)} saved, step time: {(step_time):.4f} seconds")
                    step_start = time.time()
        print(f"\nFinished. Check {self.out_path} for files.")

    def save_filename_mappings(self, path):
        """
        Save filename mappings to a csv file.

        Args:
            path: String specifying the path where to save the csv file.
        """
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['original_label_filename', 'new_subject_id'])
            for filename, subject_id in self.filename_mappings.items():
                writer.writerow([filename, subject_id])
