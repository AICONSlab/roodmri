"""
This file demonstrates how to generate a list of filenames (technically, a list
of dictionaries) for passing to the DatasetGenerator class.
"""

from pathlib import Path

"""
Consider the example dataset directory structure shown below:

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

The code below demonstrates how to generate a list of dictionaries, specifying
full paths to test data locations, suitable for passing to the DatasetGenerator
class as the input_files argument. This method makes use of the glob method for
Path objects from the pathlib library, but can also be accomplished using
a combination of glob.glob (from the glob standard library) and os.path.join.
"""

data_dir = Path('/home/user/data/')
image_paths = [str(path) for path in sorted(data_dir.glob('test_data/*/t1.nii.gz'))]
label_paths = [str(path) for path in sorted(data_dir.glob('test_data/*/seg_label.nii.gz'))]
input_files = [{'image': img, 'label': lbl} for img, lbl in zip(image_paths, label_paths)]

"""
The same directory structure with subject-specific filenames:

/home/user/data/
|-- train_data
`-- test_data
    |-- case_01
    |   |-- case_01_t1.nii.gz
    |   `-- case_01_seg_label.nii.gz
    |-- case_02
    |   |-- case_02_t1.nii.gz
    |   `-- case_02_seg_label.nii.gz
    .
    .
    .
    `-- case_99
        |-- case_99_t1.nii.gz
        `-- case_99_seg_label.nii.gz
"""

data_dir = Path('/home/user/data/')
image_paths = [str(path) for path in sorted(data_dir.glob('test_data/*/*t1.nii.gz'))]
label_paths = [str(path) for path in sorted(data_dir.glob('test_data/*/*seg_label.nii.gz'))]
input_files = [{'image': img, 'label': lbl} for img, lbl in zip(image_paths, label_paths)]

"""
And a different directory structure altogether:

/home/user/data/
|-- train_data
`-- test_data
    |-- t1_images
    |   |-- case_01_t1.nii.gz
    |   |-- case_02_t1.nii.gz
    |   .
    |   .
    |   .
    |   `-- case_99_t1.nii.gz
    `-- segmentation_labels
        |-- case_01_seg_label.nii.gz
        |-- case_02_seg_label.nii.gz
        .
        .
        .
        `-- case_99_seg_label.nii.gz
"""

data_dir = Path('/home/user/data/')
image_paths = [str(path) for path in sorted(data_dir.glob('test_data/t1_images/*.nii.gz'))]
label_paths = [str(path) for path in sorted(data_dir.glob('test_data/segmentation_labels/*.nii.gz'))]
input_files = [{'image': img, 'label': lbl} for img, lbl in zip(image_paths, label_paths)]
