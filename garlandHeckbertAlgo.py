import bpy  # Blender Python API for interacting with Blender
import numpy as np  # Numpy for numerical operations
import sys  # Add for error handeling


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
                    Q += np.matmul(p, p.T)  # Add the outer product of p to the quadric matrix
            v.Q = Q  # Assign the computed quadric matrix to the vertex

    def update_error_quadrics(self, new_vertex, affected_vertices):
        """
        Update the quadric error matrices for the new vertex and its neighbors.
        
        The new vertex's quadric matrix is the sum of the quadric matrices of the contracted vertices.
        The affected vertices are those connected to the contracted vertices; their quadric matrices 
        need to be updated based on the new geometry.
        """
        # Compute the quadric matrix for the new vertex        
        self.compute_error_quadrics([new_vertex])

        # Update the quadric matrices for the affected vertices
        self.compute_error_quadrics(affected_vertices)

    def select_valid_pairs(self):
        for edge in self.edges:
            self.pairs.add((edge.v1, edge.v2))

        for i, v1 in enumerate(self.vertices):
            for j, v2 in enumerate(self.vertices):
                if i < j and np.linalg.norm(v1.position - v2.position) < self.threshold:
                    self.pairs.add((v1, v2))


simplify = gh_mesh_simplify(0.2, 0.8)
simplify.simplify_obj()
