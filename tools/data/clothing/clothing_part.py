"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2021
"""

# External imports
import numpy as np

import torch

# from pytorch3d.io import save_obj

# Internal imports
from tools.data.three_dim_calc import rotation_matrix_from_vectors


class ClothingPart:

    """
    Clothing part objects
    """

    def __init__(self, global_vertices, global_faces, vertices, faces, uv_vertices, uv_faces):
        self.global_vertices = global_vertices
        self.global_faces = global_faces
        self.vertices = vertices
        self.faces = faces
        self.uv_vertices = uv_vertices
        self.uv_faces = uv_faces
        self.self_seams = []
        self.edges = []

        self.find_seams()

    def find_seams(self):
        """
        Find the self seams fo this clothing part. These are seams that connect two sections of the same clothing part.
        Returns:
        """
        # Create edge matrix
        uv_edge_matrix = [[[] for x in range(len(self.uv_vertices))] for y in range(len(self.uv_vertices))]

        # Get all the uv edges
        for index, face in enumerate(self.uv_faces.numpy()):
            for i in range(len(face)):
                for j in range(i, len(face)):
                    if i != j:
                        indices = [face[i], face[j]]
                        min_index = int(np.min(indices))
                        max_index = int(np.max(indices))
                        uv_edge_matrix[min_index][max_index].append((index, i, j))

        # Create edge matrix
        edge_matrix = [[[] for x in range(len(self.vertices))] for y in range(len(self.vertices))]

        # Get all the edges
        for index, face in enumerate(self.faces.numpy()):
            for i in range(len(face)):
                for j in range(i, len(face)):
                    if i != j:
                        indices = [face[i], face[j]]
                        min_index = int(np.min(indices))
                        max_index = int(np.max(indices))
                        edge_matrix[min_index][max_index].append((index, i, j))

        # For every edge check if it is on the border
        count = 0
        for i in range(len(self.uv_vertices)):
            for j in range(i, len(self.uv_vertices)):
                if len(uv_edge_matrix[i][j]) == 1:
                    count += 1
                    uv_face_index, face_vert_index_1, face_vert_index_2 = uv_edge_matrix[i][j][0]
                    vert_index_1 = int(self.faces[uv_face_index][face_vert_index_1])
                    vert_index_2 = int(self.faces[uv_face_index][face_vert_index_2])

                    indices = [vert_index_1, vert_index_2]
                    min_index = int(np.min(indices))
                    max_index = int(np.max(indices))

                    clothing_edge = ClothingEdge(min_index, max_index, i, j)
                    if len(edge_matrix[min_index][max_index]) == 2:
                        self.self_seams.append(clothing_edge)
                    elif len(edge_matrix[min_index][max_index]) == 1:
                        self.edges.append(clothing_edge)

    def get_global_index_vertex(self, local_vertex_index):
        return self.global_vertices[local_vertex_index]

    def prepare_vertices(self):
        """
        Translate UV to 3D coordinates
        Returns:
        """
        # Get means for 3d and uv data, fit line
        _mean_uv_value = torch.mean(self.uv_vertices, dim=0, keepdim=True)
        self.uv_vertices -= _mean_uv_value
        uu_uv, dd_uv, vv_uv = np.linalg.svd(self.uv_vertices)
        uv_direction = np.concatenate([vv_uv[0], np.zeros((1,))])
        _mean_value = torch.mean(self.vertices, dim=0, keepdim=True)
        original_vertices = self.vertices - _mean_value
        uu, dd, vv = np.linalg.svd(original_vertices)
        direction = vv[0]

        # get the rotation matrix from the directions
        rotation_matrix = rotation_matrix_from_vectors(direction, uv_direction)
        batched_rotation_matrix = torch.Tensor(rotation_matrix).unsqueeze(0).repeat(self.uv_vertices.shape[0], 1, 1)

        # Update vertices
        _vertices = torch.cat([self.uv_vertices, torch.zeros((self.uv_vertices.shape[0], 1))], dim=1)
        _vertices = torch.bmm(batched_rotation_matrix, _vertices.unsqueeze(2)).squeeze()

        # Scale # TODO This is an approximation due to not having detailed data
        # Get min and max points to obtain ratio in distance along principle component
        z_rotation_matrix = rotation_matrix_from_vectors(np.array([0, 0, 1]), direction)
        batched_z_rotation_matrix = torch.Tensor(z_rotation_matrix).unsqueeze(0).repeat(_vertices.shape[0], 1, 1)
        _new_distance_vertices = torch.bmm(batched_z_rotation_matrix, _vertices.unsqueeze(2)).squeeze().numpy()
        batched_z_rotation_matrix = torch.Tensor(z_rotation_matrix).unsqueeze(0).repeat(self.vertices.shape[0], 1, 1)
        _old_distance_vertices = torch.bmm(batched_z_rotation_matrix, original_vertices.unsqueeze(2)).squeeze().numpy()

        # Flip if needed
        for axis in range(3):
            max_uv_index = np.argmax(_vertices[:, axis])
            max_face_index = np.argmax(np.any((max_uv_index == self.uv_faces).numpy(), axis=1))
            old_max_vertex_z_sign = original_vertices[int(self.faces[max_face_index][0].item())][axis] > np.mean(_vertices, axis=0)[axis]
            new_max_vertex_z_sign = _vertices[int(self.uv_faces[max_face_index][0].item())][axis] > np.mean(_vertices, axis=0)[axis]
            print(original_vertices[int(self.faces[max_face_index][0].item())][axis], _vertices[int(self.uv_faces[max_face_index][0].item())][axis])
            if old_max_vertex_z_sign != new_max_vertex_z_sign:
                _vertices[:axis] *= -1
                _new_distance_vertices[:axis] *= -1

        # Get scale ratio
        z_distance_new = _new_distance_vertices[np.argmax(_new_distance_vertices[:, 2]), 2] - \
            _new_distance_vertices[np.argmin(_new_distance_vertices[:, 2]), 2]
        z_distance_old = _old_distance_vertices[np.argmax(_old_distance_vertices[:, 2]), 2] - \
            _old_distance_vertices[np.argmin(_old_distance_vertices[:, 2]), 2]
        scale_ratio = z_distance_old / z_distance_new

        for edge in self.self_seams + self.edges:
            edge.vert_index_1 = edge.uv_vert_index_1
            edge.vert_index_2 = edge.uv_vert_index_2
        self.faces = self.uv_faces

        return _vertices, scale_ratio

    def unfold(self, vertices):
        """
        Translate UV to 3D coordinates as an unfolded version
        Returns:
        """
        self.vertices = vertices - torch.mean(vertices, dim=0, keepdim=True) + torch.mean(self.vertices, dim=0, keepdim=True)
        for edge in self.self_seams + self.edges:
            edge.vert_index_1 = edge.uv_vert_index_1
            edge.vert_index_2 = edge.uv_vert_index_2
        self.faces = self.uv_faces

    def scale(self, vertices, scale_ratio):
        """
        Scale the 3D vertices
        Args:
            vertices:list of vertices (Nx3)
            scale_ratio: float scaling factor

        Returns:

        """
        batched_scale_matrix = (torch.eye(3) * scale_ratio).unsqueeze(0).repeat(vertices.shape[0], 1, 1)
        vertices = torch.bmm(batched_scale_matrix, vertices.unsqueeze(2)).squeeze()
        return vertices

    def save_obj(self, file_path):
        """
        Save into an OBJ format
        Args:
            file_path: file path where to save the obj

        Returns:

        """
        # save_obj(
        #     file_path,
        #     verts=self.vertices,
        #     faces=self.faces,
        #     verts_uvs=self.uv_vertices,
        #     faces_uvs=self.uv_faces
        # )


class ClothingEdge:

    """
    Class that contains the data of a clothing edge
    """

    def __init__(self, vert_index_1, vert_index_2, uv_vert_index_1, uv_vert_index_2):
        """
        Save the indices, make sure that vertex_1 is always the smallest index

        Args:
            vert_index_1: (int) index of a vertex of an edge in the 3d mesh
            vert_index_2: (int) index of a vertex of an edge in the 3d mesh
            uv_vert_index_1: (int) index of a vertex of an edge in the uv
            uv_vert_index_2: (int) index of a vertex of an edge in the uv
        """
        if vert_index_1 < vert_index_2:
            self.vert_index_1 = vert_index_1
            self.vert_index_2 = vert_index_2
        else:
            self.vert_index_1 = vert_index_2
            self.vert_index_2 = vert_index_1

        if uv_vert_index_1 < uv_vert_index_2:
            self.uv_vert_index_1 = uv_vert_index_1
            self.uv_vert_index_2 = uv_vert_index_2
        else:
            self.uv_vert_index_1 = uv_vert_index_2
            self.uv_vert_index_2 = uv_vert_index_1
