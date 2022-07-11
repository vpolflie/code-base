"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import os
from queue import Queue
import time

# Internal imports
from runner.base_runner import BaseRunner
from runner.builder import RUNNERS
from tools.python_utils import filter_index_meta_data
import tools.io as io_utils
from tools.print_utils import *

import config.config as cfg
import logger.logger as log
logger = log.logger


@RUNNERS.register_module(name='INFERENCE_RUNNER')
class InferenceRunner(BaseRunner):

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

        # Visualisation info
        self.visualiser = visualiser

        # Writer parameters
        self.queue = Queue()
        self.writer_functions = [io_utils.__dict__[writer_function.lower()]
                                 for writer_function in cfg.RUNNER.WRITER_METHODS]
        self.folders = [os.path.join(cfg.RUNNER.OUTPUT, folder) for folder in cfg.RUNNER.FOLDERS]
        self.writer_threads = []
        self.extensions = cfg.RUNNER.EXTENSIONS

        # Iteration info
        self.total_iterations = 0

    def initialise(self):
        """
        Initialise the runner
        Returns:

        """
        # Initialise output writer
        self.writer_threads = []
        for index in range(cfg.RUNNER.NUMBER_OF_WRITERS):
            # Create thread
            thread = io_utils.OutputWriterThread(
                self.queue, self.writer_functions, self.folders, self.extensions, index
            )
            # Start thread
            thread.start()

            # Keep thread
            self.writer_threads.append(
                thread
            )

    def run(self):
        """
            Run method will be responsible for executing a training iteration and a validation iteration
        """
        # Run over all epochs
        data_loader = self.data_loader.get_test_data_loader()
        iteration_start_time = time.time()
        batch_size = None

        # Iterate over the data loader
        iteration = 0
        meta_data = {}
        for index, (batch, meta_data) in enumerate(data_loader):

            # Run batch
            batch_size = meta_data["batch_size"]
            if batch_size > 0:
                results = self.model.inference(batch)

                self.update_queue(results, meta_data)

            # Update total iterations
            iteration += 1
            self.total_iterations += batch_size

            # Print updates
            if index % cfg.RUNNER.ITERATIONS_LOGGER_UPDATE == 0:
                iteration_run_time = time.time() - iteration_start_time
                progress = iteration / len(data_loader)
                self.log_iteration(progress, iteration_run_time, self.total_iterations, False)

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
            self.log_iteration(progress, iteration_run_time, self.total_iterations, True)

            # Visualise this iteration
            self.visualiser.visualise(
                self.total_iterations, 0, self.model.get_visualisation_data(),
                **meta_data
            )

        # Wait for queue to empty
        time.sleep(5)
        self.check_queue_empty()

        return

    def update_queue(self, results, meta_data):
        """
        Update queues for the output writers

        Args:
            results: the results that need to be written
            meta_data: meta data of the files/results in dictionary form

        Returns:

        """
        for result_index in range(meta_data["batch_size"]):
            _meta_data = filter_index_meta_data(meta_data, result_index, meta_data["stackable_keys"])
            self.queue.put(([r[result_index] for r in results], _meta_data))

    def check_queue_size(self):
        """
        Pass while the queue size is too big
        Returns:

        """
        while self.queue.qsize() > cfg.RUNNER.MAX_LENGTH_QUEUE:
            time.sleep(0.1)

    def check_queue_empty(self):
        """
        Pass while the queue size is too big
        Returns:

        """
        while self.queue.qsize() > 0:
            time.sleep(1)

    def log_iteration(self, progress, time_passed, iterations, done=False, runner_type="INFERENCE"):
        """
        Log the inference progress

        Args:
            progress: current progress
            time_passed: time passed
            iterations: total amount of data done

        Parameters:
            done: finished all iterations boolean
            runner_type: which type or runner is being run

        Returns:

        """

        string = "Data entries finished: "
        string += purple("%d" % iterations)
        string += "\t"
        # Process progress
        string += "Progress: "
        string += green('{:^40}'.format(progress_bar(40, progress)))
        string += "\tTime spent:" + green(" %.2f" % time_passed) + " seconds"

        if done:
            logger.log(string, runner_type, start="\r")
        else:
            logger.log(string, runner_type, start="\r", end="")

