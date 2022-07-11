"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
from abc import ABC
import os
import torch
import torch.nn as nn

# Internal imports
from model.model_api import ModelAPI
from tools.io.io_utils import dump_config, dump_visualisations
from tools.path_utils import create_folders, symbolic_link_force
from tools.print_utils import *

import config.config as cfg
import logger.logger as log
logger = log.logger


class BasePytorchModel(ModelAPI, ABC):

    """
        Base Pytorch Model Class that needs to be implemented.
        This class is mainly a pytorch interface between the framework and the python api structure
    """

    def __init__(self):
        """
            Base initialisation.
        """
        super().__init__()

        self.network = None
        self.network_caller = None
        self.network_optimizers = {}
        self.device = None

        # Check if GPU is available, set possible device and designated tensor
        if torch.cuda.is_available():
            possible_device = "cuda:0"
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            possible_device = "cpu"
            torch.set_default_tensor_type(torch.FloatTensor)
        self.device = torch.device(possible_device)

    def initialize_model(self):
        """
        Initialise model, this needs to be called after the build, this bit does all the CUDA + dataparallel stuff.
        """
        # If more GPU's are available use data parallel model
        if torch.cuda.device_count() > 1:
            self.network = nn.DataParallel(self.network)
            self.network_caller = self.network.module
        else:
            self.network_caller = self.network

        if hasattr(cfg.MODEL, "LOAD") and cfg.MODEL.LOAD:
            self.load_model(cfg.MODEL.LOAD)

        # Push network to device
        self.network.to(self.device)
        self.network_caller.device = self.device

    def train(self, data):
        """
            Perform a training step on the model
            :returns: a list containing all the loss scores for the batch
        """
        self.network_caller.train_state()
        _, losses = self.network.forward(data)
        self.network.optimize_step(losses)
        return [loss.cpu().detach().numpy() for loss in losses]

    def evaluate(self, data, return_output=True):
        """
            Evaluate the model by feeding data through the network but not 
            performing an optimizing step
            :returns: a list containing all the loss scores for the batch
        """
        self.network_caller.evaluation_state()
        with torch.no_grad():
            results, losses = self.network.forward(data)
        results = [result.cpu().detach().numpy() if not isinstance(result, list) else result for result in results] if return_output else None
        return results, [loss.cpu().detach().numpy() for loss in losses]

    def inference(self, data):
        """
            Perform inference on the model
            :returns: the results of feeding the data through the model
        """
        self.network_caller.inference_state()
        with torch.no_grad():
            results, _ = self.network_caller.inference(data)
        return [result.cpu().detach().numpy() if not isinstance(result, list) else result for result in results]

    def save_model(self, epoch, **kwargs):
        """
            Save Model by saving each subsequent network
        """
        # Output path
        save_directory_path = os.path.join("meta_data", "models", cfg.RUNNER.OUTPUT, str(epoch))

        # Create save folder
        create_folders(save_directory_path)
        dump_visualisations(save_directory_path, self.get_visualisation_data(), **kwargs)
        dump_config(save_directory_path, cfg.MODEL)

        # Create symbolic link to the latest epoch
        latest_path = os.path.join("meta_data", "models", cfg.RUNNER.OUTPUT, "latest")
        symbolic_link_force(str(epoch), latest_path)

        for block in self.network_caller.network_blocks:
            if isinstance(block.save_name, str):
                save_filename = '%s.pth' % block.save_name
                save_path = os.path.join(save_directory_path, save_filename)

                torch.save(block.state_dict(), save_path)

        if cfg.MODEL.SAVE_OPTIMIZER:
            for name, optimizer, _ in self.network_caller.optimizers:
                if isinstance(name, str):
                    save_filename = '%s.pth' % name
                    save_path = os.path.join(save_directory_path, save_filename)

                    torch.save(optimizer.state_dict(), save_path)

    def load_model(self, path):
        """
            Load a pretrained model checkpoint
        """
        for block in self.network_caller.network_blocks:
            if isinstance(block.save_name, str):
                load_filename = '%s.pth' % block.save_name
                load_path = os.path.join(path, load_filename)
                logger.log(green('Loading the model from %s' % load_path), "MODEL")
                state_dict = torch.load(load_path, map_location=str(self.device))

                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                block.load_state_dict(state_dict, strict=cfg.MODEL.STRICT_LOAD)

        if cfg.MODEL.LOAD_OPTIMIZER:
            for name, optimizer, scheduler in self.network_caller.optimizers:
                if isinstance(name, str):
                    load_filename = '%s.pth' % name
                    load_path = os.path.join(path, load_filename)
                    if os.path.exists(load_path):
                        logger.log(green('Loading the optimizer from %s' % load_path), "MODEL")
                        state_dict = torch.load(load_path, map_location=str(self.device))
                        optimizer.load_state_dict(state_dict)
                        scheduler.set_last_epoch(cfg.MODEL.LAST_EPOCH)
                    else:
                        logger.log(red('Path does not exist for %s' % load_path), "MODEL")

    def get_visualisation_data(self):
        """
        Get data that needs to be visualised from the network
        Returns: Return a list of VisualisationData objects

        """
        return self.network_caller.get_visualisation_data()

    def update_optimizer_learning_rate(self, update_percentage):
        """
        Update the learning rate of the optimizers

        args:
            update_percentage: how much of the learning rate update we've done
        """
        self.network_caller.update_learning_rate(update_percentage)

    def update_optimizer_warm_up(self, update_percentage):
        """
        Update the learning rate of the optimizers in terms of warm up. This means slowly building up the learning
        rate from 0 to the required value so the optimizer can figure out it's gradients

        args:
            update_percentage: how much of the learning rate update we've done
        """
        self.network_caller.update_optimizer_warm_up(update_percentage)

    def apply_stochastic_weight_averaging(self):
        """
        Update the optimizer to use stochastic weight averaging
        """
        self.network_caller.apply_stochastic_weight_averaging()

    def get_header_information(self):
        """
            Get the loss names of the network
            :returns: a list of strings
        """
        return self.network_caller.loss_names, self.network_caller.max_length, self.network_caller.comparison
