import bpy  # Blender Python API for interacting with Blender
import numpy as np  # Numpy for numerical operations
import sys  # Add for error handling
from scipy.spatial import KDTree  # Efficient nearest neighbor search
from concurrent.futures import ThreadPoolExecutor  # For multithreading

class Vertex:
    def __init__(self, position, id):
        # Position of the vertex in 3D space
        self.position = np.array(position)
        # Quadric error matrix initialized to zero
        self.Q = np.zeros((4, 4))
        # Unique ID for the vertex
        self.id = id
        # List to store references to faces this vertex is part of
        self.faces = set()
        

class Face:
    def __init__(self, vertices, id):
         # Unique ID for the face
        self.id = id
        # List of edges that make up this face
        self.vertices = vertices
        # Calculate initial plane equation parameters
        self.update_plane_equation()

    def update_plane_equation(self):
        """
        Update the plane equation parameters (a, b, c, d) for the face.

        The plane equation is derived from the face vertices.
        The normal vector (a, b, c) is normalized to ensure it is a unit vector.
        """
        v1, v2, v3 = self.vertices[0].position, self.vertices[1].position, self.vertices[2].position
        # Compute the normal vector of the plane using the cross product of two edge vectors
        normal = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(normal)  # Compute the length of the normal vector
        
        if norm == 0:
            self.is_degenerate = True  # Mark the face as degenerate if the normal length is zero
            return
        self.is_degenerate = False
        normal = normal / norm  # Normalize the normal vector to ensure it is a unit vector
        d = -np.dot(normal, v1)  # Compute the d parameter of the plane equation using one of the vertices
        self.plane_equation = np.append(normal, d)  # Store the plane equation parameters as a 4-element array

class Mesh3D:
    def __init__(self):
        # List to store all vertices
        self.vertices = []
        # List to store all faces
        self.faces = []
        # Set to store valid pairs for contraction
        self.pairs = set()
        # Dictionary to store costs for valid pairs
        self.pair_costs = {}
        # Reference to the active Blender object
        self.active_blender_object = None
        # Dictionary to store optimal positions for vertex pairs
        self.optimal_positions = {}
        # Counter to assign unique IDs to each vertex
        self.vertex_id_counter = 0
        # Counter to assign unique IDs to each face
        self.face_id_counter = 0

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


        # Create Face and Edge objects for each triangle in the Blender mesh
        for poly in mesh.polygons:
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

        # print("Uploading of the mesh is done!")

    def update_blender_mesh(self):
        """
        Update the existing Blender mesh with the simplified vertices and faces.
        """
        mesh = self.active_blender_object.data  # Get the mesh data of the active Blender object

        # Set the vertices and faces
        # Get the positions of the simplified vertices
        vertices = [vertex.position.tolist() for vertex in self.vertices]
        # Get the indices of the simplified faces
        faces = [[self.vertices.index(vertex) for vertex in face.vertices] for face in self.faces if not face.is_degenerate]

        # Update the mesh in Blender
        # Clear existing geometry
        mesh.clear_geometry()
        # Set new vertices and faces
        mesh.from_pydata(vertices, [], faces)
        # Update the mesh data
        mesh.update()
        # print("Updating of the mesh is done!")


# Quadric error metrics based error (Garland and Heckbert).
# Mesh simplification calss
class GHMeshSimplify(Mesh3D):
    def __init__(self, threshold=0.1, simplification_ratio=0.5):
        Mesh3D.__init__(self)
        if simplification_ratio > 1 or simplification_ratio <= 0:
            sys.exit('Error: simplification ratio should be in (0;1]).')
        if threshold < 0:
            sys.exit('Error: threshold should be non-negative.')
        # Distance threshold for valid pairs
        self.threshold = threshold
        # Ratio to simplify the mesh
        self.simplification_ratio = simplification_ratio

    # Function to load an OBJ file into Blender
    def load_obj(self, filepath):
        bpy.ops.wm.obj_import(filepath=filepath)

    # Function to run the simplification process
    def simplify_obj(self):
        active_object = bpy.context.active_object
        if active_object is None or active_object.type != 'MESH':
            raise ValueError("No active mesh object selected in Blender.")

        self.active_blender_object = active_object
        self.load_from_blender()
        self.initial_compute_error_quadrics()
        self.simplify()
        self.update_blender_mesh()
        print("Done!")

    def compute_vertex_Q(self, v):    
        # Initialize the quadric error matrix to zero
        Q = np.zeros((4, 4))
        for f in v.faces:
            if not f.is_degenerate:
                # Get the plane equation parameters for the face
                p = f.plane_equation
                # Reshape to a column vector
                p = p.reshape(1, len(p))
                # Add the outer product of p to the quadric matrix
                Q += np.matmul(p.T, p)
                # print("plane: ", p)
        # Assign the computed quadric matrix to the vertex
        v.Q = Q
        # print("vertex,",v.id)
        # print("Q: ", Q)
        
    def initial_compute_error_quadrics(self):
        """
        Compute the initial (multithreaded) quadric error matrices for each vertex.

        Each vertex's quadric matrix is the sum of the outer products of the plane equations 
        of the faces that include this vertex. This matrix represents the error metric for the vertex.
        """            
        with ThreadPoolExecutor() as executor:
            executor.map(self.compute_vertex_Q, self.vertices)
        # for v in self.vertices:
        #     self.compute_vertex_Q(v)

    def select_valid_pairs(self):
        """
        Select valid pairs of vertices for contraction based on edges and distance threshold.
        """
        # Add all existing edges as valid pairs
        for face in self.faces:
            if not face.is_degenerate:
                # Add edges for each face
               for i, v1 in enumerate(face.vertices):
                    v2 = face.vertices[(i + 1) % len(face.vertices)]
                    self.pairs.add(tuple(sorted((v1, v2), key=lambda vertex: vertex.id)))

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
                results = executor.map(check_valid_pairs, range(len(self.vertices)))
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
        assert(v1.id < v2.id)
        # Sum the quadric matrices of the pair
        Q = v1.Q + v2.Q
        # Modify the last row for homogeneous coordinates
        Q_new=np.concatenate([Q[:3,:], np.array([0,0,0,1]).reshape(1,4)], axis=0)
        if np.linalg.det(Q) > 0:
            # Solve Qv = [0, 0, 0, 1]
            vvvvv = np.matmul(np.linalg.inv(Q_new), np.array([0, 0, 0, 1]).reshape(4, 1))
            # Extract the optimal position
            # Compute the cost of a given position based on the quadric matrix Q.
            # print("iterm:", np.matmul(vvvvv.T, Q))
            min_cost = np.matmul(np.matmul(vvvvv.T, Q), vvvvv)
            v_optimal = vvvvv.reshape(4)[:3]
        else:
            min_cost = float('inf')
            t_values = np.linspace(0, 1, 10)
            positions = [v1.position, v2.position, (v1.position + v2.position) /2] + \
                [(1 - t) * v1.position + t * v2.position for t in t_values]
            for pos in positions:
                pos = np.append(pos, 1).reshape(4, 1)
                cost = np.matmul(np.matmul(pos.T, Q), pos)
                if cost < min_cost:
                    min_cost = cost
                    v_optimal = pos.reshape(4)[:3]
        # Store the optimal position in the dictionary
        self.optimal_positions[pair] = v_optimal
        # Compute the cost of the optimal position
        # print("matrix", Q)
        # print("points", v1.position, " ", v2.position)
        # print("cost", v_optimal, " ",min_cost)
        return min_cost

    def simplify(self):
        initial_number_vertices = len(self.vertices)
        target_number_vertices = int(initial_number_vertices * self.simplification_ratio)
        removed_number_vertices = initial_number_vertices - target_number_vertices

        self.select_valid_pairs()

        with ThreadPoolExecutor() as executor:
            pair_costs = executor.map(lambda pair: (pair, self.compute_pair_cost(pair)), self.pairs)
            self.pair_costs = {pair: cost for pair, cost in pair_costs}

        # for pair in self.pairs:
        #     self.pair_costs[pair] = self.compute_pair_cost(pair)
        currently_removed_number_vertices = 0

        while initial_number_vertices - currently_removed_number_vertices > target_number_vertices and self.pair_costs:
            v1, v2 = min(self.pair_costs, key=self.pair_costs.get)
            new_position = self.optimal_positions[(v1, v2)]

            for pair in list(self.pair_costs.keys()):
                if v2 in pair:
                    del self.pair_costs[pair]

            v1.position = new_position

            v1.faces.update(v2.faces)
            
            new_pairs = set()
            for face in list(v1.faces):
                if not face.is_degenerate:
                    for i, v in enumerate(face.vertices):
                        if v == v2:
                            face.vertices[i] = v1
                    face.update_plane_equation()
                    if face.is_degenerate:
                        v1.faces.remove(face)
                        continue
                    else:
                        num = len(face.vertices)
                        for i, v in enumerate(face.vertices):
                            v3 = face.vertices[(i + 1) % num]
                            if v != v3 and (v == v1 or v3 == v1):
                                new_pairs.add(tuple(sorted((v, v3), key=lambda vertex: vertex.id)))
                        
            self.compute_vertex_Q(v1)
                    
            for pair in new_pairs:
                self.pair_costs[pair] = self.compute_pair_cost(pair)

            self.vertices.remove(v2)

            if currently_removed_number_vertices % 100 == 0:
                percentage = 100 * currently_removed_number_vertices / removed_number_vertices
                remaining_vertices_until_done = removed_number_vertices - currently_removed_number_vertices
                print(f"{percentage:.2f}% done with {remaining_vertices_until_done} vertices remaining to be removed.")
            currently_removed_number_vertices += 1

    def load_obj_file(self):
        """
        Load a 3D model from an OBJ file. Parses vertices and faces.
        """
        self.vertices = []
        self.faces = []
        vertex_map = {}  # Map from vertex index to Vertex object

        with open("D:\\Math-Concepts-For-Devs\\00.Project\\models\\dinosaur.obj", 'r') as file:
            for line in file:
                if line.startswith("v "):
                    # Parse vertex coordinates
                    parts = line.split()
                    position = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    vertex = Vertex(position, self.vertex_id_counter)
                    self.vertices.append(vertex)
                    vertex_map[len(self.vertices)] = vertex  # OBJ indices are 1-based
                    self.vertex_id_counter += 1
                elif line.startswith("f "):
                    # Parse face indices
                    parts = line.split()
                    vertex_indices = [int(parts[1]), int(parts[2]), int(parts[3])]
                    face_vertices = [vertex_map[idx] for idx in vertex_indices]
                    face = Face(face_vertices, self.face_id_counter)
                    self.face_id_counter += 1
                    self.faces.append(face)
                    for vertex in face_vertices:
                        vertex.faces.add(face)

        # print("Loaded OBJ file from:", "D:\\Math-Concepts-For-Devs\\00.Project\\models\\dinosaur.obj")

    def output(self):
        """
        Output the simplified mesh to an OBJ file.
        """
        with open("D:\\Math-Concepts-For-Devs\\00.Project\\models\\dinosaurSimp.obj", 'w') as file:
            # Write vertices
            for vertex in self.vertices:
                file.write(f"v {vertex.position[0]} {vertex.position[1]} {vertex.position[2]}\n")

            # Write faces
            for face in self.faces:
                if not face.is_degenerate:
                    vertex_indices = [self.vertices.index(vertex) + 1 for vertex in face.vertices]  # OBJ indices are 1-based
                    file.write(f"f {vertex_indices[0]} {vertex_indices[1]} {vertex_indices[2]}\n")

        # print("Output simplified model to:", "D:\\Math-Concepts-For-Devs\\00.Project\\models\\dinosaurSimp.obj")


# Example usage:
simplify = GHMeshSimplify(0, 0.5)
simplify.simplify_obj()