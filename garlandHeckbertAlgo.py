import bpy  # Blender Python API for interacting with Blender
import numpy as np  # Numpy for numerical operations
import sys  # Add for error handling


class Vertex:
    def __init__(self, position, id):
        # Position of the vertex in 3D space
        self.position = np.array(position)
        self.Q = np.zeros((4, 4))  # Quadric error matrix initialized to zero
        self.id = id  # Unique ID for the vertex


class Edge:
    def __init__(self, v1, v2):
        self.v1 = v1  # One vertex of the edge
        self.v2 = v2  # The other vertex of the edge


class Face:
    def __init__(self, vertices):
        self.vertices = vertices  # List of vertices that make up this face
        self.update_plane_equation()  # Calculate initial plane equation parameters

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

        if norm == 0:
            self.is_degenerate = True  # Mark the face as degenerate if the normal length is zero
            return
        self.is_degenerate = False
        normal = normal / norm  # Normalize the normal vector to ensure it is a unit vector
        # Compute the d parameter of the plane equation using one of the vertices
        d = -np.dot(normal, v1)
        # Store the plane equation parameters as a 4-element array
        self.plane_equation = np.append(normal, d)


class mesh3D:
    def __init__(self):
        self.vertices = []  # List to store all vertices
        self.edges = []  # List to store all edges
        self.faces = []  # List to store all faces
        self.pairs = set()  # Set to store valid pairs for contraction
        self.pair_costs = {}  # Dictionary to store costs for valid pairs
        self.active_blender_object = None  # Reference to the active Blender object
        self.optimal_positions = {}  # Dictionary to store optimal positions for vertex pairs
        self.vertex_id_counter = 0  # Counter to assign unique IDs to each vertex

    def load_from_blender(self):
        """
        Load the mesh data from the actively selected Blender object and construct the mesh.
        """
        self.active_blender_object = bpy.context.active_object  # Get the currently selected Blender object

        if self.active_blender_object is None or self.active_blender_object.type != 'MESH':
            raise ValueError("No active mesh object selected in Blender.")

        # Access the mesh data from the Blender object
        mesh = self.active_blender_object.data
        vertex_map = {}  # Map to store Vertex objects indexed by their Blender vertex index

        # Create Vertex objects for each vertex in the Blender mesh
        for v in mesh.vertices:
            # Create a Vertex object with position and ID
            vertex = Vertex(position=np.array(v.co), id=self.vertex_id_counter)
            self.vertex_id_counter += 1  # Increment the unique ID counter
            self.vertices.append(vertex)  # Add vertex to the list
            # Map Blender vertex index to Vertex object
            vertex_map[v.index] = vertex

        # Ensure the active mesh is triangulated
        bpy.context.view_layer.objects.active = self.active_blender_object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris()  # Convert quads to triangles
        bpy.ops.object.mode_set(mode='OBJECT')

        # Create Face and Edge objects for each triangle in the Blender mesh
        for poly in mesh.polygons:
            face_vertices = [vertex_map[mesh.loops[loop_index].vertex_index]
                             for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total)]
            # Create a Face object with the vertices
            face = Face(face_vertices)
            if not face.is_degenerate:
                self.faces.append(face)  # Add face to the list
                # Add edges for each face
                num_vertices = len(face_vertices)
                for i in range(num_vertices):
                    v1 = face_vertices[i]
                    v2 = face_vertices[(i + 1) % num_vertices]
                    # Create an Edge object with two vertices
                    edge = Edge(v1, v2)
                    self.edges.append(edge)  # Add edge to the list
        print("Uploading of the mesh is done!")

    def update_blender_mesh(self):
        """
        Update the existing Blender mesh with the simplified vertices and faces.
        """
        mesh = self.active_blender_object.data  # Get the mesh data of the active Blender object

        # Set the vertices and faces
        # Get the positions of the simplified vertices
        vertices = [vertex.position.tolist() for vertex in self.vertices]
        faces = [[self.vertices.index(vertex) for vertex in face.vertices] for face in self.faces if len(
            face.vertices) == 3]  # Get the indices of the simplified faces

        # Update the mesh in Blender
        mesh.clear_geometry()  # Clear existing geometry
        mesh.from_pydata(vertices, [], faces)  # Set new vertices and faces
        mesh.update()  # Update the mesh data
        print("Updating of the mesh is done!")


# Quadric error metrics based error (Garland and Heckbert).
# Mesh simplification calss
class gh_mesh_simplify(mesh3D):
    def __init__(self, threshold=0.1, simplification_ratio=0.5):
        mesh3D.__init__(self)
        if simplification_ratio > 1 or simplification_ratio <= 0:
            sys.exit('Error: simplification ratio should be in (0;1]).')
        if threshold < 0:
            sys.exit('Error: threshold should be non-negative.')
        self.threshold = threshold  # Distance threshold for valid pairs
        self.simplification_ratio = simplification_ratio  # Ratio to simplify the mesh

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
        self.compute_error_quadrics(self.vertices)
        self.simplify()
        self.update_blender_mesh()
        print("Done!")

    def compute_error_quadrics(self, affected_vertices):
        """
        Compute the quadric error matrices for each vertex from the array affected_vertices.

        Each vertex's quadric matrix is the sum of the outer products of the plane equations 
        of the faces that include this vertex. This matrix represents the error metric for the vertex.
        """
        for v in affected_vertices:
            Q = np.zeros((4, 4))  # Initialize the quadric error matrix to zero
            for f in self.faces:
                if v in f.vertices:  # Check if the vertex is part of the face
                    p = f.plane_equation  # Get the plane equation parameters for the face
                    p = p.reshape(4, 1)  # Reshape to a column vector
                    # Add the outer product of p to the quadric matrix
                    Q += np.matmul(p, p.T)
            v.Q = Q  # Assign the computed quadric matrix to the vertex
        print("Computing the initial error quadrics is done!")

    def update_error_quadrics(self, new_vertex, affected_vertices):
        """
        Update the quadric error matrices for the new vertex and its neighbors.

        The new vertex's quadric matrix is the sum of the quadric matrices of the contracted vertices.
        The affected vertices are those connected to the contracted vertices; their quadric matrices 
        need to be updated based on the new geometry.
        """
        # Compute the quadric matrix for the new vertex and update the quadric matrices for the affected vertices
        self.compute_error_quadrics(affected_vertices.append(new_vertex))

    def select_valid_pairs(self):
        """
        Select valid pairs of vertices for contraction based on edges and distance threshold.
        """
        for edge in self.edges:
            # Add all existing edges as valid pairs
            self.pairs.add((edge.v1, edge.v2))

        for i, v1 in enumerate(self.vertices):
            for j, v2 in enumerate(self.vertices):
                # Check if distance is below the threshold
                if i < j and np.linalg.norm(v1.position - v2.position) < self.threshold:
                    self.pairs.add((v1, v2))  # Add the pair if valid
        print("Selecting the initial valid to contract pairs is done!")

    def compute_position_cost(self, Q, pos):
        """
        Compute the cost of a given position based on the quadric matrix Q.
        """
        v = np.append(
            pos, 1)  # Convert the position to homogeneous coordinates
        # Compute the cost as v^T * Q * v
        return np.matmul(v.T, np.matmul(Q, v))

    def compute_pair_cost(self, pair):
        """
        Compute the cost of contracting a pair of vertices and store the optimal position.

        The cost is computed as the minimum error introduced by contracting the pair of vertices.
        The optimal position is calculated by solving the system Qv = [0, 0, 0, 1]^T if Q is invertible.
        If Q is not invertible, alternative methods are used to find the optimal position.
        """
        v1, v2 = pair
        Q = v1.Q + v2.Q  # Sum the quadric matrices of the pair
        Q[3, :] = [0, 0, 0, 1]  # Modify the last row for homogeneous coordinates
        try:
            # Solve Qv = [0, 0, 0, 1]
            v = np.linalg.solve(Q, np.array([0, 0, 0, 1]))
            v_optimal = v[:3]  # Extract the optimal position
        except np.linalg.LinAlgError:
            best_position = None
            min_cost = float('inf')
            t_values = np.linspace(0, 1, 10)
            for t in t_values:
                pos = (1 - t) * v1.position + t * v2.position
                cost = self.compute_position_cost(Q, pos)
                if cost < min_cost:
                    min_cost = cost
                    best_position = pos

            if best_position is not None:
                v_optimal = best_position
            else:
                positions = [v1.position, v2.position,
                             (v1.position + v2.position) / 2]
                for pos in positions:
                    cost = self.compute_position_cost(Q, pos)
                    if cost < min_cost:
                        min_cost = cost
                        v_optimal = pos

        # Store the optimal position in the dictionary
        self.optimal_positions[pair] = v_optimal

        # Compute the cost of the optimal position
        return self.compute_position_cost(Q, v_optimal)

    def update_faces(self, new_vertex, removed_vertices):
        """
        Update faces by replacing removed vertices with the new vertex.
        """
        new_faces = []
        for face in self.faces:
            for i, v in enumerate(face.vertices):
                if v in removed_vertices:
                    # Replace the removed vertex with the new vertex
                    face.vertices[i] = new_vertex
            face.update_plane_equation()  # Update the plane equation parameters for the face
            if not face.is_degenerate:
                new_faces.append(face)

        self.faces = new_faces  # Remove degenerate faces

    def update_valid_pairs(self, new_vertex, removed_vertices):
        """
        Update the valid pairs after contracting vertices.
        """        
        # Remove the contracted vertices
        self.vertices = [v for v in self.vertices if v not in removed_vertices]
        new_pairs = set()

        # Check existing edges and threshold condition
        for v in self.vertices:
            if v != new_vertex:
                if (new_vertex, v) in self.edges or (v, new_vertex) in self.edges:
                    # Add new valid pairs if there is an edge
                    new_pairs.add((new_vertex, v))
                elif np.linalg.norm(new_vertex.position - v.position) < self.threshold:
                    # Add new valid pairs if the distance is below the threshold
                    new_pairs.add((new_vertex, v))

        # Remove invalid pairs involving removed vertices
        self.pairs = {
            (v1, v2) for v1, v2 in self.pairs if v1 in self.vertices and v2 in self.vertices}
        # Update the valid pairs set with the new pairs
        self.pairs = new_pairs.union(self.pairs)
        
        # Remove costs for pairs involving removed vertices
        for v in removed_vertices:
            for pair in list(self.pair_costs.keys()):
                if v in pair:
                    del self.pair_costs[pair]

        # Recalculate costs for new pairs
        for pair in new_pairs:
            self.pair_costs[pair] = self.compute_pair_cost(pair)
        
        # Add the new vertex to the list
        self.vertices.append(new_vertex)

    def simplify(self):
        """
        Simplify the mesh by contracting vertices until the target number of vertices is reached.
        """
        # Calculate the target number of vertices
        target_number_vertices = int(len(self.vertices) * self.simplification_ratio)
        # Select valid pairs based on the threshold and existing edges
        self.select_valid_pairs()
        # Initialize the pair costs
        self.pair_costs = {pair: self.compute_pair_cost(
            pair) for pair in self.pairs}
        # Counter for progress reports
        initilal_number_vertices = len(self.vertices)
        removed_number_vertices= target_number_vertices - initilal_number_vertices
        while len(self.vertices) > target_number_vertices and self.pair_costs:
            # Find the pair with the minimum cost
            v1, v2 = min(self.pair_costs, key=self.pair_costs.get)
            # Retrieve the precomputed optimal position
            new_position = self.optimal_positions[(v1, v2)]
            # Create a new vertex with a unique ID
            new_vertex = Vertex(new_position, self.vertex_id_counter)
            self.vertex_id_counter += 1

            affected_vertices = set()
            for edge in self.edges:
                if edge.v1 in [v1, v2]:
                    affected_vertices.add(edge.v2)
                if edge.v2 in [v1, v2]:
                    affected_vertices.add(edge.v1)
            affected_vertices = list(affected_vertices)

            # Update the faces to replace removed vertices with the new vertex
            self.update_faces(new_vertex, [v1, v2])
            # Update the Q matrices for the new vertex and its neighbors
            self.update_error_quadrics(
                new_vertex, affected_vertices + [new_vertex])
            # Update the valid pairs and vertex list
            self.update_valid_pairs(new_vertex, [v1, v2])
            
            if removed_number_vertices % 20 == 0:
                percentage = 100 * (initilal_number_vertices - removed_number_vertices) / (initilal_number_vertices - removed_number_vertices)
                remaining_vertices_until_done = initilal_number_vertices - (removed_number_vertices + target_number_vertices)
                print(str(percentage) + '%' + ' done with ' + str(remaining_vertices_until_done) + ' vertices remaining to be removed.')
            removed_number_vertices += 1

simplify = gh_mesh_simplify(0.2, 0.9)
simplify.simplify_obj()
