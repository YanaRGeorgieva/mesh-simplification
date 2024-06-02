import bpy
import numpy as np
import sys 

class HalfEdge:
    def __init__(self, vertex=None, next=None, twin=None, face=None):
        self.vertex = vertex  # Vertex at the end of this half-edge
        self.next = next      # Next half-edge around the face
        self.twin = twin      # Opposite half-edge
        self.face = face      # Face this half-edge belongs to


class Vertex:
    def __init__(self, position, id=None):
        self.position = position  # 3D coordinates
        self.half_edge = None     # One of the outgoing half-edges
        self.id = id            # Unique id via which we can identify it


class Face:
    def __init__(self):
        self.half_edge = None     # One of the half-edges bordering this face


class mesh3D:
    def __init__(self):
        self.vertices = []
        self.to_be_deleted_vertices = []
        self.half_edges = []
        self.to_be_deleted_half_edges = []
        self.faces = []
        self.to_be_deleted_faces = []
        self.active_blender_object = None

    def load_from_blender(self):
        """
        Load the mesh data from the actively selected Blender object and construct the Half-Edge structure.
        """
        self.active_blender_object = bpy.context.active_object  # Get the currently selected Blender object

        if self.active_blender_object is None or self.active_blender_object.type != 'MESH':
            raise ValueError("No active mesh object selected in Blender.")

        # Access the mesh data from the Blender object
        mesh = self.active_blender_object.data
        vertex_map = {}  # Map to store Vertex objects indexed by their Blender vertex index
        edge_map = {}  # Map to store Half-Edge objects for twin lookup

        # Create Vertex objects for each vertex in the Blender mesh
        for v in mesh.vertices:
            # Create a Vertex object with the position
            vertex = Vertex(position=np.array(v.co), id=v.index)
            # Add the vertex to the list of vertices
            self.vertices.append(vertex)
            # Map the Blender vertex index to the Vertex object
            vertex_map[v.index] = vertex

        # Ensure the active mesh is triangulated
        bpy.context.view_layer.objects.active = self.active_blender_object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris()
        bpy.ops.object.mode_set(mode='OBJECT')

        # Create Face and Half-Edge objects for each triangle in the mesh
        for poly in mesh.polygons:
            assert len(poly.vertices) == 3, "Mesh is not fully triangulated."

            face = Face()  # Create a Face object
            self.faces.append(face)  # Add the face to the list of faces
            prev_half_edge = None  # Previous half-edge in the face loop
            first_half_edge = None  # First half-edge in the face loop

            # Iterate over the vertices of the triangle to create Half-Edges
            for i in np.arange(3):
                v1 = poly.vertices[i]
                v2 = poly.vertices[(i + 1) % 3]
                # Get the Vertex object for the current vertex index
                vertex = vertex_map[v1]
                # Create a Half-Edge object
                half_edge = HalfEdge(vertex=vertex, face=face)

                if vertex.half_edge is None:
                    # Assign this half-edge as an outgoing half-edge of the vertex
                    vertex.half_edge = half_edge

                if prev_half_edge:
                    # Link the previous half-edge to the current half-edge
                    prev_half_edge.next = half_edge
                else:
                    first_half_edge = half_edge  # Store the first half-edge of the face loop

                prev_half_edge = half_edge  # Update the previous half-edge to the current half-edge
                # Add the half-edge to the list of half-edges
                self.half_edges.append(half_edge)

                # Create or update twin half-edge
                key = (v1, v2)
                twin_key = (v2, v1)
                if twin_key in edge_map:
                    # Get the existing twin half-edge
                    twin_half_edge = edge_map[twin_key]
                    half_edge.twin = twin_half_edge  # Link the current half-edge to its twin
                    twin_half_edge.twin = half_edge  # Link the twin half-edge to the current half-edge
                else:
                    # Store the half-edge for twin lookup
                    edge_map[key] = half_edge

            # Complete the loop for the face
            # Link the last half-edge back to the first half-edge
            prev_half_edge.next = first_half_edge
            face.half_edge = first_half_edge  # Assign the first half-edge to the face

    # def update_blender_mesh(self):
    #     """
    #     Update the Blender mesh with the modified vertices and faces.
    #     """
    #     mesh = self.active_blender_object.data  # Access the mesh data from the stored Blender object reference
        
    #     model = generate_new_model()
        
    #     # Update vertex positions
    #     for i, vertex in enumerate(model.vertices):
    #         mesh.vertices[i].co = vertex.position

    #     # Rebuild the faces based on the half-edge structure
    #     new_faces = []
    #     for face in model.faces:
    #         h_edge = face.half_edge
    #         face_indices = []
    #         for _ in range(3):
    #             face_indices.append(model.vertices.index(h_edge.vertex))
    #             h_edge = h_edge.next
    #         new_faces.append(tuple(face_indices))

    #     # Update the mesh data
    #     mesh.clear_geometry()
    #     mesh.from_pydata(
    #         [vertex.position for vertex in model.vertices], [], new_faces)
    #     mesh.update()

    #     # Refresh the Blender viewport
    #     bpy.context.view_layer.update()


# Quadric error metrics based error (Garland and Heckbert).
# Mesh simplification calss

class gh_mesh_simplify(mesh3D):
    def __init__(self, threshold, simplification_ratio):
        mesh3D.__init__(self)
        if simplification_ratio > 1 or simplification_ratio <= 0:
            sys.exit('Error: simplification ratio should be in (0;1]).')
        if threshold < 0:
            sys.exit('Error: threshold should be non-negative.')
        self.t = threshold
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
                
    def generate_valid_half_pairs(self):
        """
        Identify and generate all valid pairs of vertices that can be considered for contraction during the mesh simplification process.
        A pair is considered valid if the distance between the two vertices is less than or equal to a specified threshold t.
        """
        threshold_pairs = []

        vertices_positions = np.array(list(map(lambda v: v.position, self.vertices)))
        for vertex in self.vertices:
            vertex_position = vertex.position
            # For the current vertex, calculate the Euclidean distance to all other vertices
            distances_to_other_vertices = np.linalg.norm(vertices_positions - vertex_position, axis=1)
            # Identify the indices of vertices that are within the threshold distance t
            close_vertices_indices = np.where(distances_to_other_vertices <= self.t)[0]
            
            # Create pairs of the current vertex with each valid nearby vertex.
            for close_vertex_idx in close_vertices_indices:
                vert = self.vertices[close_vertex_idx]
                if vert != vertex:
                    # Imaginary half-edges. Added artificially.
                    v1 = HalfEdge(vertex=vertex, face=vertex.face, next=None)
                    v2 = HalfEdge(vertex=vert, face=vert.face, next=None)
                    v1.twin = v2
                    v2.twin = v1
                    threshold_pairs.append(v1)
                    threshold_pairs.append(v2)
        val_pairs=[]
        if len(threshold_pairs) > 0:
            val_pairs= self.half_edges.extend(self.threshold_pairs)
        else:
            val_pairs = self.half_edges

        # Ensure unique pairs
        _, unique_indices = np.unique(list(map(lambda edge: edge.vertex.id * (10*10) + edge.twin.vertex.id, val_pairs)), return_index=True)
        self.valid_pairs = np.array(val_pairs)[unique_indices]          

    
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
        self.generate_valid_half_pairs()
        print(self.valid_pairs)
        # self.calculate_optimal_contraction_pairs_and_cost()
        # self.iteratively_remove_least_cost_valid_pairs()        
        # self.update_blender_mesh()
        print("Done!")


simplify = gh_mesh_simplify(0.2, 1)
simplify.load_obj("D:\\Math-Concepts-For-Devs\\00.Project\\Mesh_simplification_python-master\\models\\dinosaur.obj")
simplify.simplify_obj()
