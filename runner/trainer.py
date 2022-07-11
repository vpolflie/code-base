"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import numpy as np
import time

# Internal imports
from runner.base_runner import BaseRunner
from runner.builder import RUNNERS
from tools.print_utils import *

import config.config as cfg
import logger.logger as log
logger = log.logger


@RUNNERS.register_module(name='TRAINER_RUNNER')
class TrainerRunner(BaseRunner):

    """
    Trainer runner module. This is a basic runner module where the key objective is to train a model,
    given a data loader and to visualise this training using the visualiser
    """

    def __init__(self, data_loader, model, visualiser):
        """
        Save all the input variables as class variables
        Args:
            data_loader: a data loader object
            model: a model object
            visualiser: a visualiser object
        """
        super().__init__(data_loader, model, visualiser)

        # Printer info
        self.headers, self.minimum, self.comparison = self.model.get_header_information()
        for index, comparison in enumerate(self.comparison):
            if comparison == "<":
                self.comparison[index] = lambda x, y: x < y
            else:
                self.comparison[index] = lambda x, y: x > y
        self.space = []
        self.best = np.asarray([None] * (len(self.headers) * 2 * 2)).reshape((len(self.headers), 2, 2))

        # Visualisation info
        self.visualiser = visualiser

        # Learning Rate Update Parameters
        self.total_learning_rate_update_epochs = cfg.RUNNER.EPOCHS - cfg.RUNNER.EPOCHS_NO_LEARNING_RATE_UPDATE
        self.warm_up_iterations_check = 0
        self.total_learning_rate_warm_up_iterations = cfg.MODEL.WARM_UP_ITERATIONS if hasattr(cfg.MODEL, "WARM_UP_ITERATIONS") else 0

        # Iteration info
        self.total_iterations = 0

    def run(self):
        """
            Run method will be responsible for executing a training iteration and a validation iteration
        """
        # Run over all epochs
        self.log_header()
        iteration_start_time = time.time()
        for epoch in range(cfg.RUNNER.EPOCHS):
            # Train an epoch
            last_batch_meta_data = self.iterate_data(self.data_loader.get_train_data_loader(), True, epoch)
            # Validate an epoch
            if cfg.RUNNER.VALIDATION and ((epoch % cfg.RUNNER.EVALUATE_EPOCHS) == 0 or (epoch + 1) == cfg.RUNNER.EPOCHS):
                last_batch_meta_data = self.iterate_data(self.data_loader.get_validation_data_loader(), False, epoch)

            if epoch > cfg.RUNNER.EPOCHS_NO_LEARNING_RATE_UPDATE and \
                    (epoch + 1) % cfg.RUNNER.EPOCHS_LEARNING_RATE_UPDATE == 0:
                self.model.update_optimizer_learning_rate(
                    update_percentage=(epoch - cfg.RUNNER.EPOCHS_NO_LEARNING_RATE_UPDATE) / self.total_learning_rate_update_epochs
                )

            if hasattr(cfg.RUNNER, "STOCHASTIC_WEIGHT_AVERAGING_PERIOD") and \
                    (epoch + 1) % cfg.RUNNER.STOCHASTIC_WEIGHT_AVERAGING_PERIOD == 0:
                self.model.apply_stochastic_weight_averaging()

            if (epoch + 1) % cfg.RUNNER.EPOCHS_MODEL_SAVE == 0 or epoch == cfg.RUNNER.EPOCHS - 1:
                self.model.save_model(epoch, **last_batch_meta_data)
        self.log_end(time.time() - iteration_start_time)

    def iterate_data(self, data_loader, update_model, epoch):
        """
            This method will take a data loader and a mode.
            We will iterate over the data loader and feed the data into the model.
            Depending on the mode the model will be updated.
            Args:
                data_loader: pytorch data loader
                update_model: boolean, indicates whether or not the model should be upgraded
                epoch: current epoch
            Returns:

        """
        iteration_start_time = time.time()
        batch_size = None

        # Iterate over the data loader
        iteration = 0
        total_data_entries = 0
        losses = None
        meta_data = {}
        for index, (batch, meta_data) in enumerate(data_loader):
            # Run batch
            batch_size = meta_data["batch_size"]

            # Update total iterations
            iteration += 1
            total_data_entries += batch_size

            if batch_size > 0:
                if update_model:
                    _losses = self.model.train(batch)
                else:
                    _, _losses = self.model.evaluate(batch)

                if losses is not None:
                    losses += np.array(_losses) * batch_size
                else:
                    losses = np.array(_losses) * batch_size

                # Print updates
                if index % cfg.RUNNER.ITERATIONS_LOGGER_UPDATE == 0:
                    iteration_run_time = time.time() - iteration_start_time
                    progress = iteration / len(data_loader)
                    self.log_iteration(epoch, progress, losses / total_data_entries,
                                       iteration_run_time, 1 - int(update_model), False)

                # Update total amount of iterations
                if update_model:
                    self.total_iterations += batch_size
                    self.warm_up_iterations_check += 1

                    # Warm up optimizer
                    if hasattr(cfg.MODEL, "WARM_UP") and hasattr(cfg.MODEL, "WARM_UP_ITERATIONS"):
                        if cfg.MODEL.WARM_UP and self.warm_up_iterations_check <= cfg.MODEL.WARM_UP_ITERATIONS:
                            self.model.update_optimizer_warm_up(
                                update_percentage=self.warm_up_iterations_check / self.total_learning_rate_warm_up_iterations
                            )

                # Visualise this iteration
                if (index + 1) % cfg.RUNNER.ITERATIONS_VISUALISATION_UPDATE == 0:
                    self.visualiser.visualise(
                        self.total_iterations, 1 - int(update_model),
                        self.model.get_visualisation_data(),
                        **meta_data
                    )

        if batch_size is not None:
            iteration_run_time = time.time() - iteration_start_time
            progress = iteration / len(data_loader)
            self.log_iteration(epoch, progress, losses / total_data_entries,
                               iteration_run_time, 1 - int(update_model), True)

            # Visualise this iteration
            self.visualiser.visualise(
                self.total_iterations, 1 - int(update_model),
                self.model.get_visualisation_data(),
                **meta_data
            )
        return meta_data

    # LOGGER FUNCTIONS BELOW

    def log_header(self):
        """
        Log the header information of the model
        Returns:

        """
        # Update the header information with additional non model related cetagories
        self.headers = ["  Mode  ", "Epoch", "       Progress       "] + self.headers + ["Time"]
        self.minimum = ["%s", "%d", "%s"] + self.minimum + ["%0.2f"]
        for i in range(len(self.headers)):
            try:
                max_length = max([len(self.headers[i]), len(self.minimum[i] % 10)])
            except TypeError as e:
                max_length = 25
            self.space.append(max_length)

        # Create the print message
        strings = []
        for i in range(len(self.headers)):
            strings += [('{:^' + str(self.space[i]) + '}').format(self.headers[i])]
        string = " | ".join(strings)
        string = "| " + string + " |"
        logger.log(len(string) * "-", "TRAINING")
        logger.log(string, "TRAINING")
        logger.log(len(string) * "-", "TRAINING")

    def log_iteration(self, epoch, progress, data, time_passed, mode, done):
        """
        Log a training iteration

        Args:
            epoch: current epoch
            progress: current progress through the current epoch
            data: the data that needs to be displayed
            time_passed: time passed
            mode: which mode training/validation
            done: epoch finished boolean

        Returns:

        """

        string = "| "

        # Process mode
        if mode == 0:
            mode_string = "TRAIN"
            mode_color_function = purple
        elif mode == 1:
            mode_string = "VALID"
            mode_color_function = blue
        else:
            mode_string = "UNKWN"
            mode_color_function = red

        string += mode_color_function(('{:>' + str(self.space[0]) + '}').format(
            (self.minimum[0] % mode_string))[:self.space[0]]
        )
        string += " | "

        # Process epoch
        string += ('{:>' + str(self.space[1]) + '}').format(
            (self.minimum[1] % epoch)[:self.space[1]]
        )
        string += " | "

        # Process progress
        string += ('{:^' + str(self.space[2]) + '}').format(
            (self.minimum[2] % progress_bar(self.space[2], progress))[:self.space[2]]
        )
        string += " | "

        # Print losses and metrics
        loss_color_functions = []

        # Keep track of the best score so far and color the current iteration if it exceeds the previous best
        for i in range(len(data)):
            if self.best[i][mode][0] is not None:
                if self.comparison[i](data[i], self.best[i][mode][1]):
                    if done:
                        self.best[i][mode] = (epoch, data[i])
                    loss_color_functions.append(mode_color_function)
                else:
                    loss_color_functions.append(lambda x: x)
            else:
                if done:
                    self.best[i][mode] = (epoch, data[i])
                loss_color_functions.append(mode_color_function)

        for i, d in enumerate(data):
            # update index
            index = i + 3

            # Process loss
            string += loss_color_functions[i](('{:>' + str(self.space[index]) + '}').format(
                (self.minimum[index] % d))[:self.space[index]]
            )
            string += " | "

        # Process time
        string += ('{:>' + str(self.space[-1]) + '}').format((self.minimum[-1] % time_passed)[:self.space[-1]])
        string += " |"

        if done:
            logger.log(string, "TRAINING", start="\r")
        else:
            logger.log(string, "TRAINING", start="\r", end="")

    def log_end(self, total_time):
        """
        Log the end of the training
        Args:
            total_time: total time passed in seconds
        Returns:

        """
        logger.log((len(self.space) - 1 + np.sum([i for i in self.space])) * "-", "TRAINING")
        logger.log("", "TRAINING")
        logger.log(green("Total training time: %02d:%02d:%02d" %
                   (total_time // 3600, (total_time % 3600) // 60, total_time % 60)), "TRAINING")
        for index, values in enumerate(self.best):
            string = "%s:"
            string_values = [self.headers[index + 3]]
            if cfg.RUNNER.TRAIN:
                string += purple(" best training " + self.minimum[index + 3] + " on epoch %d ")
                string_values += [self.best[index][0][1], self.best[index][0][0]]

            if cfg.RUNNER.VALIDATION:
                string += " - "
                string += blue(" best validation " + self.minimum[index + 3] + " on epoch %d")
                string_values += [self.best[index][1][1], self.best[index][1][0]]

            string = string % tuple(string_values)
            logger.log(string, "TRAINING")
