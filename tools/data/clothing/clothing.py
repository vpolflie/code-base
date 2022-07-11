"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2021
"""

# External imports
import os
from pathlib import Path
import pickle

# Internal imports
from tools.data.clothing.clothing_graph import ClothingGraph
from tools.data.clothing.trousers_graph import TrousersGraph


class Clothing:

    """
    Clothing class, this exists from multiple ClothingPart objects and a ClothingGraph
    """

    def __init__(self, clothing_parts, type=None, length=None):
        self.clothing_parts = clothing_parts

        if type is None:
            self.clothing_graph = ClothingGraph(self.clothing_parts)
        elif type == "trouser":
            self.clothing_graph = TrousersGraph(self.clothing_parts)
        else:
            self.clothing_graph = None

        self.length = length

        self.clothing_graph.unfold()

    def save_obj(self, save_path):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        for index, part in enumerate(self.clothing_parts):
            part.save_obj(os.path.join(save_path, "part_%03d.obj" % index))

    def save_pickle(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
