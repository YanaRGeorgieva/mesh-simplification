import re
import sys  # Add for error handling
from concurrent.futures import ThreadPoolExecutor  # For multithreading

import bpy  # Blender Python API for interacting with Blender
import trimesh

import numpy as np  # Numpy for numerical operations

from scipy.spatial import KDTree  # Efficient nearest neighbor search

class Vertex:
    def __init__(self, position, id):
        # Initialize the position of the vertex in 3D space
        self.position = np.array(position)
        # Initialize the quadric error matrix to zero
        self.Q = np.zeros((4, 4))
        # Unique ID for the vertex
        self.id = id
        # Set to store references to faces this vertex is part of
        self.faces = set()


class Face:
    def __init__(self, vertices, id):
        # Unique ID for the face
        self.id = id
        # List of vertices that make up this face
        self.vertices = vertices
        # By default it is not degenerate
        self.is_degenerate = False
        # Calculate initial plane equation parameters
        self.update_plane_equation()

    def calculate_normal(self) :
        """Calculate the normal of a given face."""
        v1 = self.vertices[1].position - self.vertices[0].position
        v2 = self.vertices[2].position - self.vertices[0].position
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)

        if np.isclose(norm, 0):
            # Handle degeneracy: Return a zero vector
            self.is_degenerate = True
            return np.zeros(3)

        return normal / norm

    def update_plane_equation(self):
        """
        Update the plane equation parameters (a, b, c, d) for the face.

        The plane equation is derived from the face vertices.
        The normal vector (a, b, c) is normalized to ensure it is a unit vector.
        """
        v1, v2, v3 = self.vertices[0].position, self.vertices[1].position, self.vertices[2].position
        # Compute the normal vector of the plane using the cross product of two edge vectors
        normal = np.cross(v2 - v1, v3 - v1)
        # Compute the length of the normal vector
        norm = np.linalg.norm(normal)

        if np.isclose(norm, 0):
            # Mark the face as degenerate if the normal length is zero
            self.is_degenerate = True
            return
        # Normalize the normal vector to ensure it is a unit vector
        normal = normal / norm
        # Compute the d parameter of the plane equation using one of the vertices
        d = -np.dot(normal, self.vertices[0].position)
        # Store the plane equation parameters as a 4-element array
        self.plane_equation = np.append(normal, d)


class GHMeshSimplify:
    def __init__(self, threshold=0.1, simplification_ratio=0.5, penalty_weight=2000.0):
        super().__init__()
        if simplification_ratio > 1 or simplification_ratio <= 0:
            sys.exit('Error: simplification ratio should be in (0;1]).')
        if threshold < 0:
            sys.exit('Error: threshold should be non-negative.')
        # Distance threshold for valid pairs
        self.threshold = threshold
        # Ratio to simplify the mesh
        self.simplification_ratio = simplification_ratio
        # Penalty weight for boundary or discontinuity edges
        self.penalty_weight = penalty_weight

    def is_boundary_edge(self, v1, v2):
        """
        Check if the edge (v1, v2) is a boundary edge.
        A boundary edge will have only one face associated with it.
        """
        shared_faces = v1.faces.intersection(v2.faces)
        return len(shared_faces) == 1

    def generate_perpendicular_plane(self, v1, v2):
        """
        Generate a perpendicular plane through the edge (v1, v2).
        This plane is perpendicular to the edge and passes through it.
        """
        edge_vector = v2.position - v1.position
        edge_midpoint = (v1.position + v2.position) / 2.0

        # Find a perpendicular vector to the edge. For simplicity, we can cross the edge with a random vector
        # and ensure it is not collinear with the edge vector.
        random_vector = np.array([1, 0, 0]) if not np.allclose(edge_vector, [1, 0, 0]) else np.array([0, 1, 0])
        perpendicular_vector = np.cross(edge_vector, random_vector)
        
        norm = np.linalg.norm(perpendicular_vector)
        if np.isclose(norm, 0): # I forgot to handle this here as well.
            # If the perpendicular vector's norm is close to zero, we try another "random" vector
            random_vector = np.array([0, 1, 0]) if np.allclose(edge_vector, [0, 1, 0]) else np.array([0, 0, 1])
            perpendicular_vector = np.cross(edge_vector, random_vector)
            norm = np.linalg.norm(perpendicular_vector)

            if np.isclose(norm, 0):
                # As a last resort, we perturb the edge vector slightly to generate a non-zero perpendicular vector
                perturbed_edge_vector = edge_vector + np.random.normal(0, 1e-5, size=edge_vector.shape)
                perpendicular_vector = np.cross(perturbed_edge_vector, random_vector)
                norm = np.linalg.norm(perpendicular_vector)
                
                if np.isclose(norm, 0):
                    return None

        perpendicular_vector = perpendicular_vector / norm

        # Plane equation: a*x + b*y + c*z + d = 0
        a, b, c = perpendicular_vector
        d = -np.dot(perpendicular_vector, edge_midpoint)

        return np.array([a, b, c, d])

    def add_boundary_quadrics(self, face):
        """
        Add quadrics for boundary/discontinuity edges to the vertices.
        """
        for i, v1 in enumerate(face.vertices):
            v2 = face.vertices[(i + 1) % len(face.vertices)]
            if self.is_boundary_edge(v1, v2):
                # Generate a perpendicular plane for the boundary edge
                plane = self.generate_perpendicular_plane(v1, v2)
                if plane is None:
                    continue # If nothing can be generated

                # Convert the plane to a quadric
                plane_quadric = np.outer(plane, plane)

                # Weight the quadric by the penalty weight
                weighted_quadric = self.penalty_weight * plane_quadric

                # Add the weighted quadric to the vertex quadrics
                v1.Q += weighted_quadric
                v2.Q += weighted_quadric

    def compute_vertex_Q(self, v):
        """
        Compute the initial quadric error matrix for a vertex.
        Each vertex's quadric matrix is the sum of the outer products of the plane equations
        of the faces that include this vertex. This matrix represents the error metric for the vertex.
        """
        Q = np.zeros((4, 4))  # Initialize the quadric error matrix to zero
        for f in v.faces:
            if not f.is_degenerate:
                # Get the plane equation parameters for the face
                p = f.plane_equation
                # Reshape to a column vector
                p = p.reshape(1, len(p))
                # Add the outer product of p to the quadric matrix
                Q += np.matmul(p.T, p)
        # Assign the computed quadric matrix to the vertex
        v.Q = Q

    def initial_compute_error_quadrics(self):
        """
        Compute the initial (multithreaded) quadric error matrices for each vertex.
        """
        with ThreadPoolExecutor() as executor:
            executor.map(self.compute_vertex_Q, self.vertices)

        # After computing the initial quadrics, add boundary quadrics
        with ThreadPoolExecutor() as executor:
            executor.map(self.add_boundary_quadrics, self.faces)

    def select_valid_pairs(self):
        """
        Select valid pairs of vertices for contraction based on edges and distance threshold.
        """
        # Add all existing edges as valid pairs
        for face in self.faces:
            # Add edges for each face
            for i, v1 in enumerate(face.vertices):
                v2 = face.vertices[(i + 1) % len(face.vertices)]
                self.pairs.add(
                    tuple(sorted((v1, v2), key=lambda vertex: vertex.id)))

        if self.threshold > 0:
            # Using a KD-tree to optimize the selection of points which satisfy the threshold distance
            vertex_positions = np.array([v.position for v in self.vertices])
            kdtree = KDTree(vertex_positions)

            def check_valid_pairs(i):
                valid_pairs = set()
                v1 = self.vertices[i]
                indices = kdtree.query_ball_point(v1.position, self.threshold)
                for j in indices:
                    if i < j:
                        v2 = self.vertices[j]
                        if v1.id < v2.id:
                            valid_pairs.add((v1, v2))
                return valid_pairs

            with ThreadPoolExecutor() as executor:
                results = executor.map(
                    check_valid_pairs, range(len(self.vertices)))
                for result in results:
                    self.pairs.update(result)
        print("Selecting the initial valid to contract pairs is done!")

    def compute_pair_cost(self, pair):
        """
        Compute the cost of contracting a pair of vertices and store the optimal position.

        The cost is computed as the minimum error introduced by contracting the pair of vertices.
        The optimal position is calculated by solving the system Qv = [0, 0, 0, 1]^T if Q is invertible.
        If Q is not invertible, alternative methods are used to find the optimal position.
        """
        v1, v2 = pair
        assert (v1.id < v2.id)
        # Sum the quadric matrices of the pair
        Q = v1.Q + v2.Q
        # Modify the last row for homogeneous coordinates
        Q_new = np.concatenate([Q[:3, :], np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
        if np.linalg.det(Q_new) > 1e-10:
            # Solve Qv = [0, 0, 0, 1]
            v = np.linalg.solve(Q_new, np.array([0, 0, 0, 1]))
            # Extract the optimal position
            v_optimal = v.reshape(4)[:3]
            # Compute the cost of a given position based on the quadric matrix Q
            min_cost = np.matmul(np.matmul(v.T, Q), v).item()
        else:
            min_cost = float('inf')
            # Test multiple positions along the edge
            t_values = np.linspace(0, 1, 10)
            positions = [v1.position, v2.position, (v1.position + v2.position) / 2] + \
                [(1 - t) * v1.position + t * v2.position for t in t_values]
            for pos in positions:
                pos = np.append(pos, 1).reshape(4, 1)
                cost = np.matmul(np.matmul(pos.T, Q), pos).item()
                if cost < min_cost:
                    min_cost = cost
                    v_optimal = pos.reshape(4)[:3]
        # Store the optimal position in the dictionary
        self.optimal_positions[pair] = v_optimal
        return min_cost

    def prevent_mesh_inversion(self, v1, v2):
        """
        Check if contracting this pair of vertices would cause mesh inversion.
        """
        affected_faces = [face for face in v1.faces if v2 in face.vertices]

        # Calculate normals before contraction
        original_normals = [face.calculate_normal() for face in affected_faces]

        # Simulate contraction: temporarily set v1's position to the new optimal position
        original_position_v1 = v1.position.copy()
        v1.position = self.optimal_positions[(v1, v2)]

        # Calculate normals after contraction
        new_normals = [face.calculate_normal() for face in affected_faces]

        # Revert the position of v1
        v1.position = original_position_v1

        # Check for any normal flips
        for original_normal, new_normal in zip(original_normals, new_normals):
            if np.dot(original_normal, new_normal) < 0:  # Dot product < 0 indicates a flip
                return False  # Disallow this contraction

        return True  # Allow this contraction

    def simplify(self):
        """
        Simplify the mesh by contracting vertices until the target number of vertices is reached.
        """
        initial_number_vertices = len(self.vertices)
        target_number_vertices = int(initial_number_vertices * self.simplification_ratio)
        removed_number_vertices = initial_number_vertices - target_number_vertices

        # Select valid pairs based on the threshold and existing edges
        self.select_valid_pairs()

        # Initialize the pair costs
        with ThreadPoolExecutor() as executor:
            pair_costs = executor.map(lambda pair: (pair, self.compute_pair_cost(pair)), self.pairs)
            self.pair_costs = {pair: cost for pair, cost in pair_costs}

        currently_removed_number_vertices = 0

        while initial_number_vertices - currently_removed_number_vertices > target_number_vertices and self.pair_costs:
            # Find the pair with the minimum cost
            v1, v2 = min(self.pair_costs, key=self.pair_costs.get)
            new_position = self.optimal_positions[(v1, v2)]

            # Check for potential mesh inversion
            if not self.prevent_mesh_inversion(v1, v2):
                del self.pair_costs[(v1, v2)]
                del self.optimal_positions[(v1, v2)]
                continue

            # Remove the pairs in the dictionary containing the to-be-removed vertex v2
            for pair in list(self.pair_costs.keys()):
                if v2 in pair:
                    del self.pair_costs[pair]
                    del self.optimal_positions[pair]

            # Update the position of v1 to the new position
            v1.position = new_position

            # Merge faces of v2 into v1
            v1.faces.update(v2.faces)

            # Update the new pairs to have their cost recomputed
            new_pairs = set()
            for face in list(v1.faces):
                # Rewire all faces containing v2 to involve v1
                for i, v in enumerate(face.vertices):
                    if v == v2:
                        face.vertices[i] = v1
                # Update the plane equation
                face.update_plane_equation()
                # Remove the face if it is degenerate
                if face.is_degenerate:
                    v1.faces.remove(face)
                    continue
                else:
                    # Otherwise add all pairs containing v1 to have their cost recomputed
                    num = len(face.vertices)
                    for i, v in enumerate(face.vertices):
                        v3 = face.vertices[(i + 1) % num]
                        if v != v3 and (v == v1 or v3 == v1):
                            new_pairs.add(
                                tuple(sorted((v, v3), key=lambda vertex: vertex.id)))

            # Recompute the quadric error matrix for the updated vertex v1
            self.compute_vertex_Q(v1)

            # Recompute the cost for all pairs involving v1
            for pair in new_pairs:
                self.pair_costs[pair] = self.compute_pair_cost(pair)

            # Mark v2 as removed
            v2.id = -1

            if currently_removed_number_vertices % 100 == 0:
                percentage = 100 * currently_removed_number_vertices / removed_number_vertices
                remaining_vertices_until_done = removed_number_vertices - \
                    currently_removed_number_vertices
                print(
                    f"{percentage:.2f}% done with {remaining_vertices_until_done} vertices remaining to be removed.")
            currently_removed_number_vertices += 1

        print(f"100.00% done with 0 vertices remaining to be removed.")
        
        # Clean the list of the deleted vertices
        self.vertices = [vertex for vertex in self.vertices if vertex.id != -1]
        # Clean the list of the deleted faces
        faces = set()
        for vertex in self.vertices:
            faces = faces.union(vertex.faces)
        self.faces = list(faces)


class SimplifyMesh3D(GHMeshSimplify):
    def __init__(self, threshold=0.1, simplification_ratio=0.5, penalty_weight=2000.0):
        super().__init__(threshold, simplification_ratio, penalty_weight)
        # List to store all vertices
        self.vertices = []
        # List to store all faces
        self.faces = []
        # Set to store valid pairs for contraction
        self.pairs = set()
        # Dictionary to store costs for valid pairs
        self.pair_costs = {}
        # Dictionary to store optimal positions for vertex pairs
        self.optimal_positions = {}
        # Counter to assign unique IDs to each vertex
        self.vertex_id_counter = 0
        # Counter to assign unique IDs to each face
        self.face_id_counter = 0

    def load_into_blender(self, filepath):
        """
        Load an OBJ/PLY/STL file into Blender.
        """
        file_extension = re.search(r'\.([a-zA-Z0-9]+)$', filepath).group(1).lower()

        if file_extension == "obj":
            bpy.ops.wm.obj_import(filepath=filepath)
        elif file_extension == "ply":
            bpy.ops.wm.ply_import(filepath=filepath)
        elif file_extension == "stl":
            bpy.ops.wm.stl_import(filepath=filepath)
        else:
            sys.exit(f'Error: the file type {file_extension} is not supported.')

    def simplify_obj_from_blender(self):
        """
        Run the simplification process.
        """
        active_object = bpy.context.active_object
        if active_object is None or active_object.type != 'MESH':
            raise ValueError("No active mesh object selected in Blender.")

        self.active_blender_object = active_object
        self.load_from_blender()
        self.initial_compute_error_quadrics()
        self.simplify()
        self.update_blender_mesh()
        print("Done!")

    def simplify_obj_from_file(self, input_file, output_file):
        """
        Run the simplification process.
        """
        self.load_file(input_file)
        self.initial_compute_error_quadrics()
        self.simplify()
        self.output_file(output_file)
        print("Done!")

    def load_file(self, input_file):
        """
        Load a 3D model from a file using trimesh. Parses vertices and faces.
        """
        mesh = trimesh.load(input_file, force='mesh')

        # Print mesh statistics
        # self.print_mesh_stats(mesh)

        # Map from vertex index to Vertex object
        vertex_map = {}

        for i, vertex in enumerate(mesh.vertices):
            position = vertex
            vertex_obj = Vertex(position, self.vertex_id_counter)
            self.vertices.append(vertex_obj)
            vertex_map[i] = vertex_obj
            self.vertex_id_counter += 1

        for i, face_indices in enumerate(mesh.faces):
            face_vertices = [vertex_map[idx] for idx in face_indices]
            face = Face(face_vertices, self.face_id_counter)
            self.face_id_counter += 1
            self.faces.append(face)
            for vertex in face_vertices:
                vertex.faces.add(face)

        print("Loaded 3D model from:", input_file)
        del mesh

    def output_file(self, output_file):
        """
        Output the simplified mesh to a file using trimesh.
        """
        vertices = [vertex.position for vertex in self.vertices]
        faces = []

        for face in self.faces:
            if not face.is_degenerate:
                face_indices = [self.vertices.index(vertex) for vertex in face.vertices]
                faces.append(face_indices)

        # Create a trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Print mesh statistics before saving
        # self.print_mesh_stats(mesh)

        # Export to file
        mesh.export(output_file)

        print("Output simplified model to:", output_file)

    def load_from_blender(self):
        """
        Load the mesh data from the actively selected Blender object and construct the mesh.
        """
        # Get the currently selected Blender object
        self.active_blender_object = bpy.context.active_object

        if self.active_blender_object is None or self.active_blender_object.type != 'MESH':
            raise ValueError("No active mesh object selected in Blender.")

        # Access the mesh data from the Blender object
        mesh = self.active_blender_object.data
        # Map to store Vertex objects indexed by their Blender vertex index
        vertex_map = {}

        # Ensure the active mesh is triangulated
        bpy.context.view_layer.objects.active = self.active_blender_object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris()  # Convert quads to triangles
        bpy.ops.object.mode_set(mode='OBJECT')

        # Create Vertex objects for each vertex in the Blender mesh
        for v in mesh.vertices:
            # Create a Vertex object with position and ID
            vertex = Vertex(position=np.array(v.co), id=self.vertex_id_counter)
            # Increment the unique ID counter
            self.vertex_id_counter += 1
            # Add vertex to the list
            self.vertices.append(vertex)
            # Map Blender vertex index to Vertex object
            vertex_map[v.index] = vertex

        # Create Face objects for each triangle in the Blender mesh
        for poly in mesh.polygons:
            # Get vertices for the face
            face_vertices = [vertex_map[mesh.loops[loop_index].vertex_index]
                             for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total)]
            # Create a Face object with the vertices
            face = Face(face_vertices, self.face_id_counter)
            # Increment the unique ID counter
            self.face_id_counter += 1
            if not face.is_degenerate:
                # Add face to the list
                self.faces.append(face)
                for vertex in face_vertices:
                    # Add the face to the vertex's list of faces
                    vertex.faces.add(face)

        print("Uploading of the mesh is done!")

    def update_blender_mesh(self):
        """
        Update the existing Blender mesh with the simplified vertices and faces.
        """
        # Get the mesh data of the active Blender object
        mesh = self.active_blender_object.data

        # Get the positions of the simplified vertices
        vertices = [vertex.position.tolist() for vertex in self.vertices]
        # Get the indices of the simplified faces
        faces = [[vertices.index(vertex) for vertex in face.vertices]
                 for face in self.faces if not face.is_degenerate]

        # Update the mesh in Blender
        # Clear existing geometry
        mesh.clear_geometry()
        # Set new vertices and faces
        mesh.from_pydata(vertices, [], faces)
        # Update the mesh data
        mesh.update()
        print("Updating of the mesh is done!")

    def print_mesh_stats(self, mesh):
        """
        Print out various statistics of the loaded or saved mesh using trimesh.
        """
        print("\nMesh Statistics:")
        print(f"Number of vertices: {len(mesh.vertices)}")
        print(f"Number of faces: {len(mesh.faces)}")
        print(f"Is the mesh watertight?: {mesh.is_watertight}")
        print(f"Is the mesh convex?: {mesh.is_convex}")
        print(f"Mesh volume: {mesh.volume}")
        print(f"Mesh surface area: {mesh.area}")
        print(f"Euler number: {mesh.euler_number}")
        print(f"Bounding box volume: {mesh.bounding_box_oriented.volume}")
        print(f"Center of mass: {mesh.center_mass}")
        print(f"Moment of inertia: {mesh.moment_inertia}")
        print(f"Principal axes of inertia: {mesh.principal_inertia_components}")
        print(f"Bounding box extents: {mesh.extents}")

# Example usage:
# meshy = SimplifyMesh3D(0.1, 0.5)
# simplify.simplify_obj_from_blender()
# meshy.simplify_obj_from_file("...\teapot.obj", "...\sim_teapot.obj")

