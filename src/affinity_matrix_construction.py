import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

import rtree
import trimesh
import networkx as nx

from tqdm import tqdm

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from sklearn.neighbors import BallTree

from scipy.spatial import KDTree
from scipy.sparse import csr_matrix


class AffinityMatrixConstruction:
    def __init__(self, multiplier=0.005, verbose=False, geodesic_neighbourhood="exact"):
        self.multiplier = multiplier
        self.verbose = verbose
        self.geodesic_neighbourhood = geodesic_neighbourhood

    def construct_affinity_matrix(self, mesh):
        mesh = self.preprocess_mesh(mesh)
        r = np.sqrt(self.multiplier * mesh.area)
        sdf = self.compute_sdf_batch(mesh)
        if self.geodesic_neighbourhood == 'approximate':
            neighbours = self.approximate_geodesic_distance_via_KDTree(mesh, r)
        else:
            neighbours = self.compute_geodesic_distance_via_BallTree(mesh, r)
        G = self.build_sparse_graph(mesh, neighbours, sdf)
        concavities, geodesic_distances, sdf_differences = self.compute_concavities_distances_and_sdf(
            G, neighbours, r)
        affinity_matrix = self.compute_affinity_matrix(
            mesh, concavities, geodesic_distances, sdf_differences)
        return affinity_matrix

    # There will always be a need for a mesh to be repaired before undergoing clustering which involves features of its faces as inputs. That is why we will try to repair it:
    # 1. Removes duplicate faces.
    # 2. Removes degenerate faces (e.g., faces with zero area).
    # 3. Resolves self-intersections and repairs mesh topology.
    # 4. Fixes the normals to ensure manifold edges and proper orientation.
    def preprocess_mesh(self, mesh):
        """
        Preprocess a 3D mesh by performing various operations to ensure it is optimized and valid for further use.

        This function performs the following steps:
        1. Removes duplicate faces.
        2. Removes degenerate faces (e.g., faces with zero area).
        3. Resolves self-intersections and repairs mesh topology.
        4. Fixes the normals to ensure manifold edges and proper orientation.

        Parameters:
        mesh (Mesh): The input mesh object to be preprocessed.

        Returns:
        Mesh: The preprocessed mesh, with duplicate and degenerate faces removed, topology repaired, and normals fixed.
        """
        # Step 1: Resolve self-intersections, degenerate and duplicate faces and repair topology
        mesh = mesh.process(validate=True)

        # Step 2: Fix inconsistent normals
        mesh.fix_normals(multibody=True)
        if self.verbose:
            print("Preprocessing is done!")
        return mesh

    def compute_geodesic_distance_via_BallTree(self, mesh, r):
        """
        Compute the geodesic distance between faces of a mesh using a BallTree with a custom metric.

        This function computes the geodesic neighbourhood for each face in a given mesh by:
        1. Calculating the centers of each face.
        2. Constructing a BallTree using these face centers.
        3. Using a custom distance metric to measure the distance between the face centers.

        Parameters:
        mesh (Mesh): The input mesh object containing triangular faces.
        r (float): The geodesic radius to determine neighbouring faces.

        Returns:
        list of arrays: A list where each entry contains the indices of neighbouring faces within the specified radius for each face.
        """
        def geodesic_metric(x, y):
            return np.linalg.norm(face_centers[np.int64(x[3])] - face_centers[np.int64(y[3])])

        # Prepare data for Ball Tree
        face_centers = mesh.triangles_center  # Face centers
        data = np.hstack([face_centers, np.arange(
            len(face_centers)).reshape(-1, 1)])  # Add face indices

        # Build the Ball Tree with the custom metric
        tree = BallTree(data, metric=geodesic_metric)

        neighbours = tree.query_radius(data, r)
        if self.verbose:
            print("Computing geodesic neighbours w.r.t. BallTree is done!")
        return neighbours

    # This method using KDTree is way faster than the BallTree in constructing the neighbourhoods. If I have the time I would like to see if I can tune the hyperparameter $r$ to generate neighbourhoods which include the geodesic neighbourhoods which are of interest to us.

    def approximate_geodesic_distance_via_KDTree(self, mesh, r):
        """
        Approximate the geodesic distance between faces of a mesh using a KDTree.
        It is way faster than using BallTree to capture actrual geodesic neighbours.

        This function approximates the geodesic neighbourhood for each face in a given mesh by:
        1. Calculating the centers of each face.
        2. Constructing a KDTree using these face centers.
        3. Querying neighbouring faces within a radius.

        Parameters:
        mesh (Mesh): The input mesh object containing triangular faces.
        r (float): The geodesic radius to determine neighbouring faces.

        Returns:
        list of arrays: A list where each entry contains the indices of neighbouring faces within the radius for each face.    
        """
        face_centers = mesh.triangles_center  # Face centers
        tree = KDTree(face_centers)

        neighbours = tree.query_ball_tree(tree, r)
        if self.verbose:
            print("Approximating geodesic neighbours w.r.t. KDTree is done!")
        return neighbours

    def compute_sdf_batch(self, mesh, num_rays=30, cone_angle=np.radians(120)):
        """
        Compute the Shape Diameter Function (SDF) for each face in the mesh.

        Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        num_rays (int): Number of rays to shoot inside the cone for each face.
                        Default is 30.
        cone_angle (float): Opening angle of the cone in radians.
                            Default is np.radians(120).

        Returns:
        np.ndarray: SDF values for each face in the mesh.
        """
        origins = mesh.triangles_center  # Face centers
        normals = -mesh.face_normals     # Reverse normals for ray directions
        # Initialize SDF values for all faces
        sdf_values = np.zeros(len(mesh.faces))

        for i in range(len(mesh.faces)):
            origin = origins[i]
            normal = normals[i]

            # Generate random rays within the cone
            rays = []
            for _ in range(num_rays):
                # Generate random direction within the cone

                # Limits the z-component to ensure the ray falls within the cone.
                z = np.random.uniform(np.cos(cone_angle / 2), 1)
                # Azimuthal angle (rotation around the z-axis), which is sampled from 0 to 2 * π to cover the full circle
                theta = np.random.uniform(0, 2 * np.pi)
                # Trigonometric relations
                x = np.sqrt(1 - z**2) * np.cos(theta)
                y = np.sqrt(1 - z**2) * np.sin(theta)
                # The 3D direction of the ray. It is expressed in local coordinates (aligned along the z-axis)
                direction = np.array([x, y, z])

                # Align ray with face normal

                # Generates a rotation matrix that aligns the initial vector [0, 0, 1] (z-axis) with the given normal.
                # This way, we can align the rays generated in the z-axis direction with any arbitrary direction in 3D space
                rotation_matrix = trimesh.geometry.align_vectors(
                    np.array([0, 0, 1]), normal)
                # Extract the 3x3 rotation component of the full transformation matrix
                rotation_3x3 = rotation_matrix[:3, :3]
                # Rotate the original direction vector [x, y, z] by applying the rotation matrix
                ray = rotation_3x3.dot(direction)
                rays.append(ray)

            rays = np.array(rays)  # Shape: (num_rays, 3)

            # Compute ray intersections
            hit_locations, index_ray, index_tri = mesh.ray.intersects_location(
                ray_origins=np.array([origin] * num_rays),
                ray_directions=rays
            )

            # Ensure unique intersections
            valid_indices = np.unique(index_ray, return_index=True)[
                1]  # Unique rays
            if len(valid_indices) > 0:
                valid_hit_locations = hit_locations[valid_indices]
                distances = np.linalg.norm(
                    valid_hit_locations - origin, axis=1)
                # Average distance of valid hits
                sdf_values[i] = np.mean(distances)
            else:
                sdf_values[i] = 0  # No valid hits for this face
        if self.verbose:
            print(
                "Computing the Shape Diameter Function (SDF) for each face in the mesh is done!")
        return sdf_values

    def build_sparse_graph(self, mesh, neighbours, sdf):
        """
        Build a sparse graph representation of the mesh using geodesic distances, dihedral angles, and SDF differences as edge weights.

        This function constructs a graph where:
        1. Each node represents a face of the input mesh.
        2. Edges between nodes represent the neighbourhood relationships between mesh faces.
        3. Edge weights are computed using the following properties:
        - Geodesic distance (approximated by Euclidean distance between face centers).
        - Dihedral angle between neighbouring faces.
        - Absolute difference in Signed Distance Function (SDF) values between faces.

        Parameters:
        mesh (Mesh): The input mesh object containing triangular faces.
        neighbours (array like object): Adjacency list of neighbours for each face.
        sdf (array-like): An array containing the SDF value for each face of the mesh.

        Returns:
        networkx.Graph: A sparse graph representation of the mesh, with edges weighted by geodesic distance, dihedral angle, and SDF difference.
        """
        G = nx.Graph()
        face_normals = mesh.face_normals  # Face normals
        face_centers = mesh.triangles_center  # Face centers

        def compute_dihedral_angle(face_a, face_b):
            normal_a = face_normals[face_a]
            normal_b = face_normals[face_b]
            dihedral_angle = np.arccos(
                np.clip(np.dot(normal_a, normal_b), -1.0, 1.0))
            concavity_weight = np.min([dihedral_angle / np.pi, 1])
            return concavity_weight

        def process_face(i):
            edges = []
            for j in neighbours[i]:
                if i != j:  # Skip self-loops
                    # Geodesic distance (approximated as Euclidean distance)
                    geodesic_distance = np.linalg.norm(
                        face_centers[i] - face_centers[j])
                    dihedral = compute_dihedral_angle(i, j)
                    sdf_difference = np.abs(sdf[i] - sdf[j])
                    edges.append((i, j, geodesic_distance,
                                  dihedral, sdf_difference))
            return edges

        if self.verbose:
            on_list = tqdm(range(len(neighbours)),
                           desc="Building sparse graph")
        else:
            on_list = range(len(neighbours))

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            results = executor.map(process_face, on_list)

        # Add results to the graph
        for edges in results:
            for i, j, geodesic_distance, dihedral, sdf_difference in edges:
                G.add_edge(i, j, geodesic_distance=geodesic_distance,
                           dihedral_angle=dihedral, sdf_difference=sdf_difference)
        if self.verbose:
            print("Building of a sparse graph with geodesic distances, dihedral angles and absolute SDF diffrences as weights is done!")
        return G

    def compute_average_geodesic_distance_and_paths(self, G, neighbours, r):
        """
        Compute the average geodesic distance and shortest paths in a graph.

        Parameters:
        G (networkx.Graph): The input graph.
        neighbours (array like object): Adjacency list of neighbours for each face.
        r (float): The radius threshold for geodesic distance.

        Returns:
        float: The average geodesic distance.
        dict: A dictionary of shortest paths for all nodes within radius `r`.
        """
        def compute_distances_for_node_and_paths(i, nbrs):
            # Restrict to the local subgraph with relevant nodes
            local_subgraph = G.subgraph(np.append(nbrs, i))

            distances, paths = nx.single_source_dijkstra(
                local_subgraph, source=i, cutoff=r, weight="geodesic_distance"
            )
            return i, distances, paths

        # Parallelize computation over nodes
        all_distances = []
        geodesic_distances_by_face = {}
        shortest_paths_by_faces = {}

        if self.verbose:
            on_list = tqdm(range(len(neighbours)),
                           desc="Computing geodesic distances")
        else:
            on_list = range(len(neighbours))

        with ThreadPoolExecutor() as executor:
            results = executor.map(
                lambda i: compute_distances_for_node_and_paths(i, neighbours[i]), on_list)

        # Aggregate distances and paths
        for face, distances, paths in results:
            all_distances.extend(distances.values())
            geodesic_distances_by_face[face] = distances
            shortest_paths_by_faces[face] = paths

        # Compute average geodesic distance
        avg_geodesic_distance = np.mean(all_distances)

        if self.verbose:
            print("Computation of average geodesic distance is done!")

        return avg_geodesic_distance, geodesic_distances_by_face, shortest_paths_by_faces

    def compute_concavities_distances_and_sdf(self, G, neighbours, r):
        """
        Computation of concavities, geodesic distances, and SDF differences between neighbouring faces.

        Parameters:
        G (networkx.Graph): The mesh represented as a graph.
        neighbours (array like object): Adjacency list of neighbours for each face.
        r (float): Radius threshold for geodesic distance.

        Returns:
        concavities, geodesic_distances, sdf_differences: Dictionaries containing computed metrics.
        """
        concavities = defaultdict(float)
        geodesic_distances = defaultdict(float)
        sdf_differences = defaultdict(float)

        # Compute average geodesic distance and shortest paths
        avg_geodesic_distance, geodesic_distances_by_face, shortest_paths_by_face = self.compute_average_geodesic_distance_and_paths(
            G, neighbours, r)

        def process_neighbour(i):
            local_concavities = {}
            local_geodesic_distances = {}
            local_sdf_differences = {}

            for j in shortest_paths_by_face[i].keys():
                if i == j:
                    continue

                # Retrieve the path and calculate metrics along the edges
                path = shortest_paths_by_face[i][j]
                edges = list(zip(path[:-1], path[1:]))

                local_geodesic_distances[(
                    i, j)] = geodesic_distances_by_face[i][j] / avg_geodesic_distance
                local_concavities[(i, j)] = np.sum(np.fromiter(
                    (G[u][v]["dihedral_angle"] for u, v in edges), dtype=np.float64))
                local_sdf_differences[(i, j)] = np.sum(np.fromiter(
                    (G[u][v]["sdf_difference"] for u, v in edges), dtype=np.float64))

            return local_concavities, local_geodesic_distances, local_sdf_differences

        if self.verbose:
            on_list = tqdm(range(len(neighbours)),
                           desc="Computing concavities, distances and sdf")
        else:
            on_list = range(len(neighbours))

        # Parallelize computation over faces
        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda i: process_neighbour(i), on_list)

        # Aggregate results
        for conc, dist, sdf_diff in results:
            concavities.update(conc)
            geodesic_distances.update(dist)
            sdf_differences.update(sdf_diff)

        if self.verbose:
            print(
                "Computation of concavities, geodesic distances, and SDF differences is done!")

        return concavities, geodesic_distances, sdf_differences

    def compute_affinity_matrix(self, mesh, concavities, geodesic_distances, sdf_differences):
        """
        Compute an affinity matrix for the mesh faces using normalized geodesic distances, concavities, and SDF differences as weights.

        This function computes an affinity matrix representing the relationships between the mesh faces.
        The affinity weights are computed based on the following features:
        1. Geodesic distances between neighboring mesh faces.
        2. Differences in Shape Diameter Function (SDF) values between faces.
        3. Concavities between mesh faces, represented by dihedral angles.

        The computed weights are normalized and combined using Gaussian kernels to create a final affinity matrix.

        Parameters:
        mesh (Mesh): The input mesh object containing triangular faces.
        concavities (dict): A dictionary where keys are tuples of face indices and values represent the concavity between the corresponding faces.
        geodesic_distances (dict): A dictionary where keys are tuples of face indices and values represent the geodesic distances between the corresponding faces.
        sdf_differences (dict): A dictionary where keys are tuples of face indices and values represent the differences in SDF values between the corresponding faces.

        Returns:
        scipy.sparse.csr_matrix: A sparse affinity matrix where each element represents the affinity between two mesh faces.
        """
        # Normalize weights for geodesic distances, SDF differences, and concavities
        sigma1 = max(geodesic_distances.values())
        sigma2 = max(sdf_differences.values())
        sigma3 = max(concavities.values())

        # Ensure sigma values are non-zero
        sigma1 = max(sigma1, 1e-8)
        sigma2 = max(sigma2, 1e-8)
        sigma3 = max(sigma3, 1e-8)

        weights = {}
        for (i, j) in geodesic_distances.keys():

            # Normalize and clip values
            g = np.clip(geodesic_distances[(i, j)] / sigma1, -10, 10)
            s = np.clip(sdf_differences[(i, j)] / sigma2, -10, 10)
            c = np.clip(concavities[(i, j)] / sigma3, -10, 10)

            # Compute weight with division by 2 in the exponent
            weight = np.exp(-0.5 * g**2) * np.exp(-0.5 * s**2) * \
                np.exp(-0.5 * c**2)

            # Apply threshold to avoid overly small weights
            if weight > 1e-8:
                weights[(i, j)] = weight

        # Convert weights to a sparse matrix
        rows, cols, data = zip(*[(i, j, w) for (i, j), w in weights.items()])
        weight_matrix = csr_matrix((data, (rows, cols)), shape=(
            len(mesh.faces), len(mesh.faces)))
        if self.verbose:
            print("Computation of affinity matrix is done!")

        return weight_matrix
