"""
Author: Vincent Polfliet
Institute: GIM
Year: 2021
"""

# External imports
import numpy as np
import time

# Internal imports
from runner.inference.inference import InferenceRunner
from runner.builder import RUNNERS
from tools.print_utils import *

import config.config as cfg
import logger.logger as log

logger = log.logger


@RUNNERS.register_module(name='EVALUATION_RUNNER')
class EvaluationRunner(InferenceRunner):
    """
        Evaluation runner module. This is a basic runner module where the key objective is to evaluate a model,
        given a data loader. All results should be saved.
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
        self.headers, self.minimum, self.comparison = self.model.get_header_information()

    def run(self):
        """
            Run method will be responsible for executing a training iteration and a validation iteration
        """
        # Run over all epochs
        data_loader = self.data_loader.get_validation_data_loader()
        iteration_start_time = time.time()
        batch_size = None

        # Iterate over the data loader
        iteration = 0
        losses = None
        meta_data = {}
        for index, (batch, meta_data) in enumerate(data_loader):

            # Run batch
            batch_size = meta_data["batch_size"]
            if batch_size > 0:
                results, _losses = self.model.evaluate(batch, cfg.RUNNER.SAVE_RESULTS)

                if cfg.RUNNER.SAVE_RESULTS:
                    self.update_queue(results, meta_data)

                if losses is not None:
                    losses += np.array(_losses) * batch_size
                else:
                    losses = np.array(_losses) * batch_size

            # Update total iterations
            iteration += 1
            self.total_iterations += batch_size

            # Print updates
            if index % cfg.RUNNER.ITERATIONS_LOGGER_UPDATE == 0:
                iteration_run_time = time.time() - iteration_start_time
                progress = iteration / len(data_loader)
                self.log_iteration(progress, iteration_run_time, self.total_iterations, False, runner_type="EVALUATION")

            # Visualise this iteration
            if (index + 1) % cfg.RUNNER.ITERATIONS_VISUALISATION_UPDATE == 0:
                self.visualiser.visualise(
                    self.total_iterations, 0, self.model.get_visualisation_data(),
                    **meta_data
                )

            # Sleep if the queue if full to avoid running out of RAM
            self.check_queue_size()

        if batch_size is not None:
            iteration_run_time = time.time() - iteration_start_time
            progress = iteration / len(data_loader)
            self.log_iteration(progress, iteration_run_time, self.total_iterations, True, runner_type="EVALUATION")

            # Visualise this iteration
            self.visualiser.visualise(
                self.total_iterations, 0, self.model.get_visualisation_data(),
                **meta_data
            )

        # Log the total loss values
        self.log_losses(losses / self.total_iterations)

        # Wait for queue to empty
        time.sleep(5)
        self.check_queue_empty()

        return

    def log_losses(self, losses):
        """
        Log the inference progress

        Args:
            losses: current losses values

        Returns:

        """

        string = "\n\n"
        string += green("Loss Results:")
        string += "\n"
        for index, loss in enumerate(losses):
            string += blue(("%s: " + self.minimum[index]) % (self.headers[index], loss)) + "\n"
        logger.log(string, "EVALUATION")
