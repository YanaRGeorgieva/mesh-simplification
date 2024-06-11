import bpy  # Blender Python API for interacting with Blender
import numpy as np  # Numpy for numerical operations
import sys # Add for error handeling
class Vertex:
    def __init__(self, position, id):
        self.position = np.array(position)  # Position of the vertex in 3D space
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
        norm = np.linalg.norm(normal)  # Compute the length of the normal vector

        if norm == 0:
            self.is_degenerate = True  # Mark the face as degenerate if the normal length is zero
            return
        self.is_degenerate = False
        normal = normal / norm  # Normalize the normal vector to ensure it is a unit vector
        d = -np.dot(normal, v1)  # Compute the d parameter of the plane equation using one of the vertices
        self.plane_equation = np.append(normal, d)  # Store the plane equation parameters as a 4-element array


class mesh3D:
    def __init__(self, threshold=0.1, simplify_ratio=0.5):
        self.vertices = []  # List to store all vertices
        self.edges = []  # List to store all edges
        self.faces = []  # List to store all faces
        self.pairs = set()  # Set to store valid pairs for contraction
        self.pair_costs = {}  # Dictionary to store costs for valid pairs
        self.threshold = threshold  # Distance threshold for valid pairs
        self.simplify_ratio = simplify_ratio  # Ratio to simplify the mesh
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
            vertex = Vertex(position=np.array(v.co), id=self.vertex_id_counter)  # Create a Vertex object with position and ID
            self.vertex_id_counter += 1  # Increment the unique ID counter
            self.vertices.append(vertex)  # Add vertex to the list
            vertex_map[v.index] = vertex  # Map Blender vertex index to Vertex object

        # Ensure the active mesh is triangulated
        bpy.context.view_layer.objects.active = self.active_blender_object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris()  # Convert quads to triangles
        bpy.ops.object.mode_set(mode='OBJECT')

        # Create Face and Edge objects for each triangle in the Blender mesh
        for poly in mesh.polygons:
            face_vertices = [vertex_map[mesh.loops[loop_index].vertex_index] for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total)]
            face = Face(face_vertices)  # Create a Face object with the vertices
            if not face.is_degenerate:
                self.faces.append(face)  # Add face to the list
                # Add edges for each face
                num_vertices = len(face_vertices)
                for i in range(num_vertices):
                    v1 = face_vertices[i]
                    v2 = face_vertices[(i + 1) % num_vertices]
                    edge = Edge(v1, v2)  # Create an Edge object with two vertices
                    self.edges.append(edge)  # Add edge to the list
                    
    def update_blender_mesh(self):
        """
        Update the existing Blender mesh with the simplified vertices and faces.
        """
        mesh = self.active_blender_object.data  # Get the mesh data of the active Blender object
        
        # Set the vertices and faces
        vertices = [vertex.position.tolist() for vertex in self.vertices]  # Get the positions of the simplified vertices
        faces = [[self.vertices.index(vertex) for vertex in face.vertices] for face in self.faces if len(face.vertices) == 3]  # Get the indices of the simplified faces
        
        # Update the mesh in Blender
        mesh.clear_geometry()  # Clear existing geometry
        mesh.from_pydata(vertices, [], faces)  # Set new vertices and faces
        mesh.update()  # Update the mesh data


# Quadric error metrics based error (Garland and Heckbert).
# Mesh simplification calss

class gh_mesh_simplify(mesh3D):
    def __init__(self, threshold, simplification_ratio):
        mesh3D.__init__(self)
        if simplification_ratio > 1 or simplification_ratio <= 0:
            sys.exit('Error: simplification ratio should be in (0;1]).')
        if threshold < 0:
            sys.exit('Error: threshold should be non-negative.')
        self.threshold = threshold
        self.simplification_ratio = simplification_ratio
        
    def initialization_simplification(self):
        """
        Calculate the plane equations and the initial values for the Q-matrices.
        """
        self.calculate_plane_equations()
        self.calculate_Q_matrices()
        
    def calculate_plane_equations(self):
        """
        Calculate the plane equation parameters for each face.
        Each face's plane is represented by the equation ax + by + cz + d = 0.
        """
        self.plane_equation_parameters = []
        for face in self.faces:
            h_edge = face.half_edge  # Start with one of the half-edges of the face
            p1 = h_edge.vertex.position  # Position of the first vertex
            p2 = h_edge.next.vertex.position  # Position of the second vertex
            p3 = h_edge.next.next.vertex.position  # Position of the third vertex

            # Compute the normal vector of the plane (perpendicular to the face)
            normal = np.cross(p2 - p1, p3 - p1)
            # Normalize the vector to make it a unit vector
            normal /= np.linalg.norm(normal)

            # Compute the d parameter of the plane equation
            d = -np.dot(normal, p1)

            # Store the plane equation parameters (a, b, c, d)
            self.plane_equation_parameters.append(np.append(normal, d))  # Add it as a row

        # Convert the list of plane parameters to a numpy array for easier manipulation
        self.plane_equation_parameters = np.array(self.plane_equation_parameters)

    def calculate_Q_matrices(self):
        """
        Calculate the Q matrices for each vertex.
        The Q matrix for a vertex represents the sum of squared distances to the planes of its adjacent faces.
        """
        self.Q_matrices = [np.zeros((4, 4)) for _ in range(
            len(self.vertices))]  # Initialize Q matrices with zeros

        # Iterate over each face to accumulate the plane quadrics for its vertices
        for i, face in enumerate(self.faces):
            # Get the plane parameters for the current face and reshape to a column vector
            p = self.plane_equation_parameters[i].reshape(4, 1)
            # Compute the outer product to get the plane quadric matrix (4, 4)
            K_p = p @ p.T

            h_edge = face.half_edge  # Start with one of the half-edges of the face
            for _ in range(3):  # Iterate over the three vertices of the face
                # Find the index of the current vertex
                v_idx = self.vertices.index(h_edge.vertex)
                # Accumulate the quadric matrix for the vertex
                self.Q_matrices[v_idx] += K_p
                h_edge = h_edge.next  # Move to the next half-edge
                
    def generate_initial_valid_pairs(self):
        """
        Identify and generate all valid pairs of vertices that can be considered for contraction during the mesh simplification process.
        A pair is considered valid if the distance between the two vertices is less than or equal to a specified threshold t.
        """
        threshold_pairs = []

        vertices_positions = np.array(list(map(lambda v: v.position, self.vertices)))
        for vertex in self.vertices:
            # For the current vertex, calculate the Euclidean distance to all other vertices
            distances_to_other_vertices = np.linalg.norm(vertices_positions - vertex.position, axis=1)
            # Identify the indices of vertices that are within the threshold distance t
            close_vertices_indices = np.where(distances_to_other_vertices <= self.threshold)[0]
            
            # Create pairs of the current vertex with each valid nearby vertex.
            for close_vertex_idx in close_vertices_indices:
                vert = self.vertices[close_vertex_idx]
                if vert != vertex and (vertex, vert) not in threshold_pairs:
                    threshold_pairs.append((vertex, vert))

        val_pairs=[]
        edges = list(map(lambda x: (x.vertex, x.twin.vertex), self.half_edges))
        if len(threshold_pairs) > 0:
            val_pairs=  edges + threshold_pairs
        else:
            val_pairs = edges

        # Ensure unique pairs unoriented
        pair_ids = set()
        unique_valid_pairs = []
        for pair in val_pairs:
            id_pair = tuple(sorted([pair[0].id, pair[1].id]))
            if id_pair not in pair_ids:
                pair_ids.add(id_pair)
                unique_valid_pairs.append(pair)
        self.valid_pairs = unique_valid_pairs        

    def calculate_optimal_contraction_pairs_and_cost(self):
        """
        calculate the optimal contraction target (optimal_vertex) for each valid pair of vertices (v1, v2) in the mesh.
        The error associated with contracting each pair is computed using quadric matrices,
        and the pairs are then sorted by their contraction cost.
        The method identifies the pair with the minimum cost and sets it as the first pair to be contracted.
        """
        self.optimal_vertex = []
        self.optimal_vertex_cost = []
        
        for edge in self.valid_pairs:
            v1 = edge[0]
            v2 = edge[1]
            
            Q = self.Q_matrices[self.vertices.index(v1)] + self.Q_matrices[self.vertices.index(v2)]
            
            # The new row [0, 0, 0, 1] facilitates the homogeneous coordinate operations, making it possible to solve for the optimal contraction vertex using linear algebra methods.
            Q_new = np.concatenate([Q[:3, :], np.array([[0, 0, 0, 1]])], axis=0)
            
            if np.linalg.det(Q_new) > 0:
                current_best_v = np.linalg.inv(Q_new) @ np.array([0, 0, 0, 1])
                current_best_cost = current_best_v.T @ Q @ current_best_v
                current_best_v = current_best_v[:3]
            else:
                v1_pos = np.append(v1.position, 1)
                v2_pos = np.append(v2.position, 1)
                v_mid = (v1_pos + v2_pos) / 2

                cost_v1 = (v1_pos.T @ Q) @ v1_pos
                cost_v2 = (v2_pos.T @ Q) @ v2_pos
                cost_v_mid = (v_mid.T @ Q) @ v_mid

                current_best_cost = min(cost_v1, cost_v2, cost_v_mid)
                if current_best_cost == cost_v1:
                    current_best_v = v1_pos[:3]
                elif current_best_cost == cost_v2:
                    current_best_v = v2_pos[:3]
                else:
                    current_best_v = v_mid[:3]

            self.optimal_vertex.append(current_best_v)
            self.optimal_vertex_cost.append(current_best_cost)

        self.optimal_vertex = np.array(self.optimal_vertex)
        self.optimal_vertex_cost = np.array(self.optimal_vertex_cost)

    def get_half_edges_around_face(self, face):
        """
        Get all half-edges surrounding a given face.
        """
        half_edges = []
        h_edge = face.half_edge
        while True:
            half_edges.append(h_edge)
            h_edge = h_edge.next
            if h_edge == face.half_edge:
                break
        return half_edges
    
    def update_valid_pairs_and_remove_vertex(self, modified_vertex, removed_vertex):
        """
        Update valid pairs by adding new edges and removing invalid ones based on modified vertices.
        """
        # Remove pairs involving modified vertices
        self.valid_pairs = {pair for pair in self.valid_pairs if ((pair[0].id != modified_vertex.id) and (pair[1].id != modified_vertex.id)) 
                            and ((pair[0].id != removed_vertex.id) and (pair[1].id != removed_vertex.id)) }

        
        # Remove removed_vertex from vertices list
        self.vertices.remove(removed_vertex)
            
        # Add new valid pairs for modified vertices
        for v2 in self.vertices:
            if modified_vertex == v2:
                continue
            if np.linalg.norm(modified_vertex.position - v2.position) < self.threshold:
                self.valid_pairs.add((modified_vertex, v2))
                    
        # Remove duplicate pairs
        pair_ids = set()
        unique_valid_pairs = []
        for pair in self.valid_pairs:
            id_pair = tuple(sorted([pair[0].id, pair[1].id]))
            if id_pair not in pair_ids:
                pair_ids.add(id_pair)
                unique_valid_pairs.append(pair)
        self.valid_pairs = unique_valid_pairs
    
    def iteratively_remove_least_cost_valid_pairs(self):
        """
        Iteratively remove the pair of least cost, contract the pair, and update the costs.
        """
        target_vertex_count = int(self.simplification_ratio * len(self.vertices))
        
        while len(self.vertices) > target_vertex_count:
            idx = np.argmin(self.optimal_vertex_cost)
            edge_to_contract = self.valid_pairs[idx]
            new_vertex_pos = self.optimal_vertex[idx]

            v1 = edge_to_contract[0]
            v2 = edge_to_contract[1]

             # Find indices of vertices in the vertices list
            v1_index = self.vertices.index(v1)
            v2_index = self.vertices.index(v2)

            # Update vertex positions
            v1.position = new_vertex_pos
            
            # Update half-edges and faces
            for he in self.half_edges:
                if he.vertex == v2:
                    he.vertex = v1

            # Update Q-matrices for the contracted vertices
            self.update_Q([v2_index], v1_index)

            # Remove faces with degenerate edges
            degenerate_faces = [face for face in self.faces if len(set(he.vertex for he in self.get_half_edges_around_face(face))) != 3]
            for face in degenerate_faces:
                self.faces.remove(face)

            # Update plane equation parameters for remaining faces
            face_indices_to_update = [i for i, face in enumerate(self.faces) if v1 in [he.vertex for he in self.get_half_edges_around_face(face)]]
            self.update_plane_equation_parameters(face_indices_to_update)

            # Recompute valid pairs and costs
            self.update_valid_pairs_and_remove_vertex(v1, v2)
            self.calculate_optimal_contraction_pairs_and_cost()
                       

    def get_half_edges_around_face(self, face):
        """
        Get all half-edges surrounding a given face.
        """
        half_edges = []
        he = face.half_edge
        while True:
            half_edges.append(he)
            he = he.next
            if he == face.half_edge:
                break
        return half_edges

    def update_plane_equation_parameters(self, need_updating_loc):
        """
        Update plane equation parameters for the given face indices.
        """
        for i in need_updating_loc:
            face = self.faces[i]
            he = face.half_edge
            p1 = he.vertex.position
            p2 = he.next.vertex.position
            p3 = he.next.next.vertex.position
            
            normal = np.cross(p2 - p1, p3 - p1)
            normal /= np.linalg.norm(normal)
            d = -np.dot(normal, p1)
            
            self.plane_equation_parameters[i] = np.append(normal, d)

    def update_Q(self, replace_locs, target_loc):
        """
        Update the Q matrices for the contracted vertices.
        """
        target_vertex = self.vertices[target_loc]
    
        # Calculate the new Q matrix for the target vertex
        Q_temp = np.zeros((4, 4))
        
        # Include the Q matrices of the vertices being replaced
        for loc in replace_locs:
            Q_temp += self.Q_matrices[loc]
        
        # Add the Q matrix of the target vertex
        Q_temp += self.Q_matrices[target_loc]
        
        # Update the Q matrix for the target vertex
        self.Q_matrices[target_loc] = Q_temp
        
        # Update Q matrix for faces around the target vertex
        face_indices = [i for i, face in enumerate(self.faces) if target_vertex in [he.vertex for he in self.get_half_edges_around_face(face)]]
        for i in face_indices:
            p = self.plane_equation_parameters[i]
            p = p.reshape(4, 1)
            Q_temp += np.matmul(p, p.T)
        
        self.Q_matrices[target_loc] = Q_temp

    def update_valid_pairs_optimal_vertex_and_cost(self, target_loc):
        """
        Update the valid pairs, optimal vertices, and costs after contraction.
        """
        self.valid_pairs = []
        self.optimal_vertex = []
        self.cost = []
        
        target_vertex = self.vertices[target_loc]
        for he in self.half_edges:
            if he.vertex == target_vertex or he.twin.vertex == target_vertex:
                if he.twin is not None and he not in self.valid_pairs and he.twin not in self.valid_pairs:
                    self.valid_pairs.append(he)
        
        self.calculate_optimal_contraction_pairs_and_cost()
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
        self.initialization_simplification()
        self.generate_initial_valid_pairs()
        self.calculate_optimal_contraction_pairs_and_cost()
        self.iteratively_remove_least_cost_valid_pairs()        
        self.update_blender_mesh()
        print("Done!")


simplify = gh_mesh_simplify(0.2, 0.8)
simplify.simplify_obj()
