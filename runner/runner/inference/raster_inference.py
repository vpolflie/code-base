"""
Author: Vincent Polfliet
Institute: GIM
Year: 2021
"""

# External imports
import numpy as np
from queue import Queue
import time

# Internal imports
from runner.inference.inference import InferenceRunner
from runner.builder import RUNNERS
import tools.io as io_utils
from tools.python_utils import filter_index_meta_data

import config.config as cfg
import logger.logger as log
logger = log.logger


@RUNNERS.register_module(name='RASTER_INFERENCE_RUNNER')
class RasterInferenceRunner(InferenceRunner):

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

        self.queue = [Queue() for _ in range(cfg.RUNNER.NUMBER_OF_WRITERS)]
        self.assigned_queues = [[i] for i in range(cfg.RUNNER.NUMBER_OF_WRITERS)]

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
                self.queue[index], self.writer_functions, self.folders, self.extensions, index
            )
            # Start thread
            thread.start()

            # Keep thread
            self.writer_threads.append(
                thread
            )

    def update_queue(self, results, meta_data):
        """
        Update queues for the output writers

        Args:
            results: the results that need to be written
            meta_data: meta data of the files/results in dictionary form
        Returns:

        """
        for result_index in range(len(meta_data["file_index"])):
            # Get correct queue
            queue_index = None
            for _queue_index, assigned_indices in enumerate(self.assigned_queues):
                if meta_data["file_index"][result_index] in assigned_indices:
                    queue_index = _queue_index

            if queue_index is None:
                minimum_queue = np.argmin([queue.qsize() for queue in self.queue])
                queue_index = minimum_queue
                self.assigned_queues[minimum_queue].append(meta_data["file_index"][result_index])

            # Push to queue
            _meta_data = filter_index_meta_data(meta_data, result_index, meta_data["stackable_keys"])
            self.queue[queue_index].put(([r[result_index] for r in results], _meta_data))

    def check_queue_size(self):
        """
        Pass while the queue size is too big
        Returns:

        """
        while np.sum([queue.qsize() for queue in self.queue]) > cfg.RUNNER.MAX_LENGTH_QUEUE:
            pass

    def check_queue_empty(self):
        """
        Pass while the queue size is too big
        Returns:

        """
        while np.sum([queue.qsize() for queue in self.queue]) > 0:
            time.sleep(1)
