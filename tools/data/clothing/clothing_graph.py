"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2021
"""

# External imports
import torch

# Internal imports


class ClothingGraph:

    """
    Clothing graph which describes how ClothingParts are connected to forma full clothing item.
    """

    def __init__(self, list_of_clothing_parts):
        self.list_of_clothing_parts = list_of_clothing_parts

        # Create graph
        self.nodes = []
        for part in list_of_clothing_parts:
            self.nodes.append(ClothingNode(part))

        for index_1, node_1 in enumerate(self.nodes):
            for index_2, node_2 in enumerate(self.nodes[index_1 + 1:]):
                if node_1 != node_2:
                    node_1.add_neighbours(node_2)

    def unfold(self):
        """
        Translate UV to 3D coordinates as an unfolded version
        Returns:
        """
        # Get the scale ratios
        scale_ratios = []
        vertices_list = []
        for node in self.nodes:
            _vertices, _scale_ratio = node.clothing_part.prepare_vertices()
            scale_ratios.append(_scale_ratio)
            vertices_list.append(_vertices)

            for connection in node.connections:
                connection.unfold()

        # Scale
        max_scale = max(scale_ratios)
        for index, node in enumerate(self.nodes):
            vertices = node.clothing_part.scale(vertices_list[index], max_scale)
            node.clothing_part.unfold(vertices)


class ClothingNode:

    """
    Clothing Node containing information about the connection between his and his neigbours
    """

    def __init__(self, clothing_part):
        self.clothing_part = clothing_part
        self.neighbours = []
        self.connections = []

        # Setup original x, y, z min/max
        self.x_max, self.x_min = torch.max(self.clothing_part.vertices[:, 0]), torch.min(self.clothing_part.vertices[:, 0])
        self.y_max, self.y_min = torch.max(self.clothing_part.vertices[:, 1]), torch.min(self.clothing_part.vertices[:, 1])
        self.z_max, self.z_min = torch.max(self.clothing_part.vertices[:, 2]), torch.min(self.clothing_part.vertices[:, 2])

    def add_neighbours(self, node):
        """
        Add a neighbour if there are overlapping vertices

        Args:
            node: a second ClothingNode

        Returns:

        """
        vertex_pairs = []
        uv_vertex_pairs = []

        # Get all the global vertex indices
        self_vertex_indices = []
        self_vertex_indices_check = []
        for edge in self.clothing_part.edges:
            self_vert_index_1 = self.clothing_part.get_global_index_vertex(edge.vert_index_1)
            if self_vert_index_1 not in self_vertex_indices_check:
                self_vertex_indices_check.append(self_vert_index_1)
                self_vertex_indices.append((self_vert_index_1, edge.vert_index_1, edge.uv_vert_index_1))
            self_vert_index_2 = self.clothing_part.get_global_index_vertex(edge.vert_index_2)
            if self_vert_index_2 not in self_vertex_indices_check:
                self_vertex_indices_check.append(self_vert_index_2)
                self_vertex_indices.append((self_vert_index_2, edge.vert_index_2, edge.uv_vert_index_2))

        node_vertex_indices = []
        node_vertex_indices_check = []
        for edge in node.clothing_part.edges:
            node_vert_index_1 = node.clothing_part.get_global_index_vertex(edge.vert_index_1)
            if node_vert_index_1 not in node_vertex_indices_check:
                node_vertex_indices_check.append(node_vert_index_1)
                node_vertex_indices.append((node_vert_index_1, edge.vert_index_1, edge.uv_vert_index_1))
            node_vert_index_2 = node.clothing_part.get_global_index_vertex(edge.vert_index_2)
            if node_vert_index_2 not in node_vertex_indices_check:
                node_vertex_indices_check.append(node_vert_index_2)
                node_vertex_indices.append((node_vert_index_2, edge.vert_index_2, edge.uv_vert_index_2))

        # Find vertex pairs
        for self_vertex_index, local_self_vertex_index, local_uv_self_vertex_index in self_vertex_indices:
            for node_vertex_index, local_node_vertex_index, local_uv_node_vertex_index in node_vertex_indices:
                if self_vertex_index == node_vertex_index:
                    vertex_pairs.append((local_self_vertex_index, local_node_vertex_index))
                    uv_vertex_pairs.append((local_uv_self_vertex_index, local_uv_node_vertex_index))

        if vertex_pairs:
            connection = ClothingNodeConnection(self, node, vertex_pairs, uv_vertex_pairs)
            self.neighbours.append(node)
            self.connections.append(connection)
            node.neighbours.append(self)
            node.connections.append(connection)


class ClothingNodeConnection:
    """
    Clothing Node Connection containing information how two nodes are connected
    """

    def __init__(self, node_1, node_2, vertex_pairs, uv_vertex_pairs):
        self.node_1 = node_1
        self.node_2 = node_2
        self.vertex_pairs = vertex_pairs
        self.uv_vertex_pairs = uv_vertex_pairs

    def unfold(self):
        """
        Unfold to the UV settings
        Returns:

        """
        self.vertex_pairs = self.uv_vertex_pairs
