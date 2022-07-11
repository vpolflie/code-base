"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2021
"""

# External imports
import numpy as np

# Internal imports
from .clothing_graph import ClothingGraph


class TrousersGraph(ClothingGraph):

    """
    Clothing graph which describes how ClothingParts are connected to forma full clothing item.
    """

    def __init__(self, list_of_clothing_parts):
        super().__init__(list_of_clothing_parts)
        # Create specific lists
        self.legs = []
        self.inner_pockets = []
        self.outer_pockets = []
        self.misc = []

        # Identify parts
        self.identify_legs()
        self.identify_pockets()
        self.identify_misc()

        print(self.legs)
        print(self.inner_pockets)
        print(self.outer_pockets)
        print(self.misc)

    def identify_legs(self):
        """
        Identify the legs of the trouser, currently a basic method which chooses them the size of the UV mapping
        Returns:

        """
        sizes = []
        for index, node in enumerate(self.nodes):
            uv_verts = node.clothing_part.uv_vertices
            min_x = np.min(uv_verts[:, 0].numpy())
            min_y = np.min(uv_verts[:, 1].numpy())
            max_x = np.max(uv_verts[:, 0].numpy())
            max_y = np.max(uv_verts[:, 1].numpy())
            surface = (max_x - min_x) * (max_y - min_y)
            if len(sizes) < 4:
                sizes.append((index, surface))
            else:
                sizes = sorted(sizes, key=lambda x: x[1])
                if sizes[0][1] < surface:
                    sizes[0] = (index, surface)

        for index, _ in sizes:
            self.legs.append(self.nodes[index])

    def identify_pockets(self):
        """

        Returns:

        """
        pass

    def identify_misc(self):
        """

        Returns:

        """
        for node in self.nodes:
            if node not in (self.legs + self.inner_pockets + self.outer_pockets):
                self.misc.append(node)
