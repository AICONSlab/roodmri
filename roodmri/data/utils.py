def is_file_list(input_files):
    """Used to check whether input_files are formatted correctly."""
    assert isinstance(input_files, list), "input_files must be a list."
    assert len(input_files) != 0, "input_files should be a non-empty list."
    assert all(isinstance(item, dict) for item in input_files), \
        "Each item in input_files should be a dictionary."
    for item in input_files:
        assert isinstance(item, dict), \
            "Each item in input_files should be a dictionary."
        assert len(item) == 2 \
            and all([x in item.keys() for x in ['image', 'label']]), \
                ("Each dictionary in input_files should have two keys: "
                 "'image', and 'label'.")
        assert all([isinstance(x, str) for x in item.values()]), \
            "All values in the dictionaries in input_files should be strings."
        return True