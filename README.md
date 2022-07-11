# AI Code Base

Repository containing an AI code base for the INNO team. 
This repository is designed for quick development and easy retraining of already existing models.
The intended user of the repository should have an advanced knowledge or working knowledge base of AI.

## Structure

### Running and Config

The repository is run using the following command:

> python main.py --arg-1 arg_1 --arg-2 arg_2 ...

The main high level configuration can be found in "config.yaml" and should look as followed:

> data_loader: "data_loader_config" \
> model: "model_config" \
> runner: "runner_config" \
> visualiser: "visualiser_config" \

What the main function does is create a config file using high level config yaml.
The main variables in this config will point to other config files with the following structure:

<pre>
config 
  | 
  |----- data_loader 
  |           | 
  |           |------- parser_update.py 
  |           |------- yaml 
  |                      | 
  |                      |------- config_1.yaml 
  |                      |------- config_2.yaml 
  |                      |------- ... 
  |----- model 
  |           | 
  |           |------- parser_update.py 
  |           |------- yaml 
  |                      | 
  |                      |------- config_1.yaml 
  |                      |------- config_2.yaml 
  |                      |------- ... 
  ... 
</pre>

Notice that each of these folders also contain a parser_update.py file.
This file can contain an "update_parser_{config_file_name}" function which will then
allow you to add commandline arguments to the main function on a modular basis.

IMPORTANT: don't initialise any default arguments for the argument parser since these will always overwrite the config file.

When all these configs are loaded they can easily be accessed by importing the main
"config.py" file as followed:

> import config.config as cfg \
> print(cfg.MODULE_1.VARIABLE_1)

### Data Sets

For this framework we assume each data set has one of the following tree structures: 
- A coco inspired structure: The files are physically split over the "train", "validation" and "test" folders.
- An explicit split structure: Train and validation files are in the "train" folder, however a "split.json" file is available denoting which file belongs to which split.
- A random seed structure: Train and validation files are in the "train" folder, and the split configuration is defined in the data set config file. 

<pre>
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
</pre>

To easily load and combine data sets, one should define their data set in a config yaml file that should be found in:

> config/data_set_configuration/yaml/{config_file_name}.yaml

An example of such a file can be seen here for the ISPRS Potsdam data set:

<pre>
# This is the is a configuration file to the ISPRS_potsdam data set:
# https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/
# Run the following scripts to get the necessary folder structure:
#
# bash scripts/ISPRS_potsdam/
# python scripts/ISPRS_potsdam/unpack_raster_data.py

path: "data/potsdam"
structure: coco
seed:
split:
extensions: [".png", ".npy"]
file_types: ["images", "segmentations"]
</pre>

A config file thus first contains general information about the data set followed by how to get this data set in the specified structure and
finally configuration variables such as the structure and in case of a "Seed based format" als a seed and split variable.

### Data Loaders

A data loader is defined by a config file in "config/data_loader/yaml", an example can be seen below:

<pre>
# Modules
data_loader_module: "pytorch.base_pytorch_data_loader"
data_set_module: "pytorch.image_segmentation_data_set"

# Data Set Configurations
data_sets_train: ["ISPRS_potsdam"]
data_sets_validation: ["ISPRS_potsdam"]
data_sets_test: ["ISPRS_potsdam"]

# Parameters
batch_size: 16
workers: 12
</pre>

A data loader config file contains 3 different parameters. The first two concern the module that should be used
to load the data set configurations. Currently, two libraries are available for loading data:

- Pytorch
- Keras

Both have a similar interface that can easily be implemented. 

Now two modules have to be defined in the data loader config. The first is the library specific data loader and creates python "generators" to iterate over. 
The second module is a data set, this is a class that inherits from a specific class in the chosen library and allows you
to load your data in a separate thread while training your model. 

The following variables are where you define which data set configurations should be loaded for which split ("train", "validation" or "test").
The last set of variables are specific parameters such as batch size, etc.

Now one final component of the data loader can be found in the pipeline module. These are all common operations that can
be used on the data. These include color based data augmentations, spatially based data augmentations, data type transformations, ...

### Models

Currently, two deep learning library interfaces are provided.

- Keras / Tensorflow
- Pytorch

Both implementations follow a similar structure where 3 abstract classes are provided which need your implementation
in case you want to create your own model.

#### Model

The model class offers a common api over all libraries, this allows the runner to run any deep learning model without any issues.
All you need to do is create a subclass of this model where you create your own initialisation.

#### Network

The network class contains most of the higher order functionality  of your network. This is the place where you define
your blocks, optimizers, loss functions, the order of block execution, etc.

This part basically contains all the logic of your network except the actual layers.

#### Block

The final and lowest level part are the blocks. Each block initialises a group of layers, and a definition of how these 
layers should be executed. 

#### Example: GAN

In case you want to implement a GAN network you should implement the following classes.
- a generator block (layers)
- a discriminator block (layers)
- a GAN network (optimizers, losses, etc)
- a GAN model (Its only task is to initialise the GAN network)


### Runners

The key role of a runner is to combine the model, data loader and visualiser into a working script. Again the config files
for the runner classes can be found in "config/runner/yaml". Each runner contains a run function, where
the run function contain the main logic for the runner. Example runners are: training a model, performing inference on a model, ...

### Visualisers

Two libraries are currently implemented:

- Visdom
- Tensorboard

Both libaries offer the functionality to visualise:

- line graphs (losses)
- histograms
- an image
- a group of images

You can visualise all these things by calling the visualise function and feeding it a list of VisualisationData objects.

### Logger

A simple logger class which allows yo uto both write to the std out and a logger file. You can print any string by 
calling the log function with the following arguments:

- String, the actual string that needs to be printed
- Phase, a string representing the current phase of you execution
- Start = "", how the string should start, useful if you want to overwrite the previous line with \r
- End = "\n", how the string should end. Default is a newline

The most noticeable parameter is probably the phase one. The logger uses this one to keep track of the current phase of
the execution. When you log a new message with a different phase, the logger will change the phase internally and print
out a new header.

### Tools

This folder contains several utility functions:
- data_utils.py: contains utility function for data, for example converting to a certain data type, changing channel first to channel last, ...
- exceptions.py: folder to keep our custom exceptions
- io_utils.py: contains functions to load/save data from/to disk 
- keras_utils.py: contains keras specific utility functions
- path_utils.py: contains functions to do path based operations in you OS
- python_utils: contains python specific functions like for example: dynamically loading modules
- pytorch_utils.py: contains pytorch specific utility functions

### Scripts

This folder contains separate stand-alone scripts which should be run only once. These scripts can multi-purpose, from data preprocessing
to singular visualisations, data transformations, etc.

### Notebooks

This folder contains separate stand alone jupyter-notebooks which can be multi-purpose. For example: data exploration, 
result visualisation, etc.

### Data
This is a folder which is excluded from the git repository and can be a symbolic link to your specfic location on your
OS where you store your data sets.

### Meta Data
This is a folder which is excluded from the git repository and can be a symbolic link to your specfic location on your
OS where you want to store your saved models.

## Known Bugs

- Pytorch data loader can hang when working in CPU only mode. 
  This can be fixed by running "ulimit -n 5000000" in the terminal first. 
  Another solution is setting workers to 0 however this will significantly slow down the data loading
  
- Don't visualise too much since this can eat up your RAM

# Install

User the pip or conda files to install your environment:

> conda env create -f conda_environment.yaml \
> pip install -r requirements.txt
