"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External Imports
import glob
import json
import os
import random

# Internal imports
from tools.path_utils import verify_path_exists


class DataSetConfiguration:
    """
    Class that collects all the files into a single data set according to a certain structure.
    All files need to have to following naming convention: %08d.extension

    All the files will be collected in these variables: train_file_paths, validation_file_paths, test_file_paths
    These are dictionaries with the key being a specific name preferably something descriptive like "images" and the
    value being a list of paths.

    A data set can have one of three structures:

    CoCo based format:
    ------------------

    path
     |
     |---- train
     |       |----- files_type_1
     |       |           |----- file_1
     |       |           |----- file_2
     |       |           |----- ...
     |       |           |----- file_n
     |       |----- file_type_2
     |                   |----- ...
     |
     |---- validation
     |       |----- files_type_1
     |       |           |----- file_1
     |       |           |----- file_2
     |       |           |----- ...
     |       |           |----- file_m
     |       |----- file_type_2
     |                   |----- ...
     |
     |---- test
     |       |----- files_type_1
     |       |           |----- file_1
     |       |           |----- file_2
     |       |           |----- ...
     |       |           |----- file_k
     |       |----- file_type_2
     |                   |----- ...

     Or

    Split Json based format:
    ------------------------

     path
     |
     |---- split.json
     |
     |---- train
     |       |----- files_type_1
     |       |           |----- file_1
     |       |           |----- file_2
     |       |           |----- ...
     |       |           |----- file_n
     |       |----- file_type_2
     |                   |----- ...
     |
     |---- test
     |       |----- files_type_1
     |       |           |----- file_1
     |       |           |----- file_2
     |       |           |----- ...
     |       |           |----- file_k
     |       |----- file_type_2
     |                   |----- ...

     Or

    Seed based format:
    ------------------

     path
     |
     |---- train
     |       |----- files_type_1
     |       |           |----- file_1
     |       |           |----- file_2
     |       |           |----- ...
     |       |           |----- file_n
     |       |----- file_type_2
     |                   |----- ...
     |
     |---- test
     |       |----- files_type_1
     |       |           |----- file_1
     |       |           |----- file_2
     |       |           |----- ...
     |       |           |----- file_k
     |       |----- file_type_2
     |                   |----- ...

    """

    def __init__(self, config, name):
        """
        Initialise data configuration

        Args:
            config: a dictionary containing all the information
        """
        # Set variables
        self.config = config
        self.config["name"] = name
        self.name = name
        self.data_set_path = config["path"]
        self.structure = config["structure"]
        self.additional_information = config["additional_information"] if config["additional_information"] else ""

        # Get all the files
        self.train_file_paths = {}
        self.validation_file_paths = {}
        self.test_file_paths = {}

        for extension, file_type in zip(self.config["extensions"], self.config["file_types"]):
            self.train_file_paths[file_type] = \
                sorted(glob.glob(os.path.join(self.data_set_path, "train", file_type, "*" + extension)))

            if self.structure == "coco":
                self.validation_file_paths[file_type] = \
                    sorted(glob.glob(os.path.join(self.data_set_path, "validation", file_type, "*" + extension)))

            self.test_file_paths[file_type] = \
                sorted(glob.glob(os.path.join(self.data_set_path, "test", file_type, "*" + extension)))

        # Check integrity of framework
        # Cant allow zero files
        if not self.train_file_paths and not self.validation_file_paths and not self.test_file_paths:
            raise ValueError("No files found!")

        # There needs to be the same amount of files per file type
        for dictionary in [self.train_file_paths, self.validation_file_paths]:
            if dictionary:
                number_of_files = [len(v) for v in dictionary.values()]
                assert all(number_of_files[0] == n_files for n_files in number_of_files), \
                    f"Need same number over different file types! - %s" % number_of_files

        # If the data set isn't according to the coco structure, create the validation set using a split
        if self.structure != "coco":
            train_paths = list(zip(self.train_file_paths.values()))
            train_samples = None
            validation_samples = None

            # Create a random split using a random seed
            if self.structure == "seed":
                if self.config["seed"]:
                    random.seed(self.config["seed"])
                else:
                    random.seed(random.randint(0, 10000))

                samples = random.sample(range(0, len(train_paths)), len(train_paths))

                if self.config["split"]:
                    split = self.config["split"]
                else:
                    split = 0.9

                train_samples = samples[:int(split * len(samples))]
                validation_samples = samples[int(split * len(samples)):]

            # Load the json containing the split of the data and split the data accordingly
            if self.structure == "json":
                with open(os.path.join(self.data_set_path, "split.json"), 'r') as f:
                    samples = json.load(f)

                train_samples = samples["train"]
                validation_samples = samples["validation"]

            # Assign the splits
            train_samples = list(zip(*train_samples))
            validation_samples = list(zip(*validation_samples))
            self.validation_file_paths = \
                {key: validation_samples[index] for index, key in enumerate(self.train_file_paths.keys())}
            self.train_file_paths = \
                {key: train_samples[index] for index, key in enumerate(self.train_file_paths.keys())}

        # Verify all paths exist
        for dictionary in [self.train_file_paths, self.validation_file_paths, self.test_file_paths]:
            for file_paths in dictionary.values():
                for file_path in file_paths:
                    verify_path_exists(file_path)
