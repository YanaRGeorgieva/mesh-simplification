import os
import sys
import tempfile
import shutil

import numpy as np

import trimesh

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from garland_heckbert_algorithm import SimplifyMesh3D, Vertex, Face

def test_vertex_initialization():
    position = [1.0, 2.0, 3.0]
    vertex_id = 1
    vertex = Vertex(position, vertex_id)
    assert np.array_equal(vertex.position, np.array(position))
    assert vertex.id == vertex_id
    assert np.array_equal(vertex.Q, np.zeros((4, 4)))
    print("test_vertex_initialization passed.")
    
def test_face_initialization():
    v1 = Vertex([0, 0, 0], 1)
    v2 = Vertex([1, 0, 0], 2)
    v3 = Vertex([0, 1, 0], 3)
    face = Face([v1, v2, v3], 1)
    expected_plane_equation = [0, 0, 1, 0]  # Assuming the vertices form a right triangle on the XY plane
    assert np.allclose(face.plane_equation[:3], expected_plane_equation[:3])
    assert np.isclose(face.plane_equation[3], expected_plane_equation[3])
    print("test_face_initialization passed.")

def test_degenerate_face():
    v1 = Vertex([0, 0, 0], 1)
    v2 = Vertex([1, 0, 0], 2)
    v3 = Vertex([2, 0, 0], 3)  # Collinear with v1 and v2
    face = Face([v1, v2, v3], 1)
    assert face.is_degenerate
    print("test_degenerate_face passed.")

def test_compute_vertex_Q():
    v1 = Vertex([0, 0, 0], 1)
    v2 = Vertex([1, 0, 0], 2)
    v3 = Vertex([0, 1, 0], 3)
    face = Face([v1, v2, v3], 1)
    v1.faces.add(face)
    v2.faces.add(face)
    v3.faces.add(face)
    
    simplifier = SimplifyMesh3D()
    simplifier.compute_vertex_Q(v1)
    
    # The quadric should be the outer product of the plane equation
    expected_Q = np.outer(face.plane_equation, face.plane_equation)
    assert np.allclose(v1.Q, expected_Q)
    print("test_compute_vertex_Q passed.")

def test_compute_pair_cost():
    v1 = Vertex([0, 0, 0], 1)
    v2 = Vertex([1, 0, 0], 2)
    v3 = Vertex([0, 1, 0], 3)
    face = Face([v1, v2, v3], 1)
    v1.faces.add(face)
    v2.faces.add(face)
    v3.faces.add(face)

    simplifier = SimplifyMesh3D()
    simplifier.vertices = [v1, v2, v3]
    simplifier.compute_vertex_Q(v1)
    simplifier.compute_vertex_Q(v2)

    cost = simplifier.compute_pair_cost((v1, v2))
    assert cost >= 0
    print("test_compute_pair_cost passed.")

def test_is_boundary_edge():
    v1 = Vertex([0, 0, 0], 1)
    v2 = Vertex([1, 0, 0], 2)
    v3 = Vertex([0, 1, 0], 3)
    v4 = Vertex([1, 1, 0], 4)
    
    face1 = Face([v1, v2, v3], 1)
    face2 = Face([v2, v3, v4], 2)
    v1.faces.add(face1)
    v2.faces.add(face1)
    v2.faces.add(face2)
    v3.faces.add(face1)
    v3.faces.add(face2)
    v4.faces.add(face2)

    simplifier = SimplifyMesh3D()
    assert simplifier.is_boundary_edge(v1, v2)
    assert not simplifier.is_boundary_edge(v2, v3)
    print("test_is_boundary_edge passed.")

def test_mesh_simplification():
    simplifier = SimplifyMesh3D(threshold=1.0, simplification_ratio=0.5)

    # Create a simple mesh: a square made of two triangles
    v1 = Vertex([0, 0, 0], 1)
    v2 = Vertex([1, 0, 0], 2)
    v3 = Vertex([1, 1, 0], 3)
    v4 = Vertex([0, 1, 0], 4)

    face1 = Face([v1, v2, v3], 1)
    face2 = Face([v1, v3, v4], 2)

    v1.faces.update([face1, face2])
    v2.faces.add(face1)
    v3.faces.update([face1, face2])
    v4.faces.add(face2)

    simplifier.vertices = [v1, v2, v3, v4]
    simplifier.faces = [face1, face2]

    simplifier.initial_compute_error_quadrics()
    simplifier.simplify()

    # After simplification, the mesh should have fewer vertices and faces
    assert len(simplifier.vertices) < 4
    assert len(simplifier.faces) < 2
    print("test_mesh_simplification passed.")

def create_test_mesh(filepath):
    """
    Creates a more complex 3D mesh (a tetrahedron) and saves it to the specified file path.
    
    The mesh will be a 3D pyramid with a triangular base (tetrahedron).
    """
    # Define vertices of a tetrahedron
    vertices = [
        [0.0, 0.0, 0.0],  # Vertex 0 (Base)
        [1.0, 0.0, 0.0],  # Vertex 1 (Base)
        [0.5, 1.0, 0.0],  # Vertex 2 (Base)
        [0.5, 0.5, 1.0],  # Vertex 3 (Apex)
    ]

    # Define faces using the indices of the vertices
    faces = [
        [0, 1, 2],  # Base triangle
        [0, 1, 3],  # Side triangle 1
        [1, 2, 3],  # Side triangle 2
        [2, 0, 3],  # Side triangle 3
    ]

    # Create a Trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Export the mesh to the specified file path
    mesh.export(filepath)

    print(f"Test mesh created and saved to {filepath}")

def test_load_and_save_mesh(tmpdir):
    input_file = os.path.join(tmpdir, "input.obj")
    output_file = os.path.join(tmpdir, "output.obj")
    
    create_test_mesh(input_file)
    
    simplifier = SimplifyMesh3D()
    simplifier.load_file(input_file)
    simplifier.output_file(output_file)
    
    assert os.path.exists(output_file)
    print("test_load_and_save_mesh passed.")

def test_remove_degenerate_faces():
    simplifier = SimplifyMesh3D()

    # Create vertices that form a degenerate face (all collinear)
    v1 = Vertex([0, 0, 0], 1)
    v2 = Vertex([1, 0, 0], 2)
    v3 = Vertex([2, 0, 0], 3)

    face = Face([v1, v2, v3], 1)

    v1.faces.add(face)
    v2.faces.add(face)
    v3.faces.add(face)

    simplifier.vertices = [v1, v2, v3]
    simplifier.faces = [face]

    simplifier.simplify()

    # The degenerate face should be removed
    assert len(simplifier.faces) == 0
    print("test_remove_degenerate_faces passed.")

def run_tests():
    # Create a temporary directory for testing
    tmpdir = tempfile.mkdtemp()

    try:
        # Run all test functions
        test_vertex_initialization()
        test_face_initialization()
        test_degenerate_face()
        test_compute_vertex_Q()
        test_compute_pair_cost()
        test_is_boundary_edge()
        test_mesh_simplification()
        test_load_and_save_mesh(tmpdir)
        test_remove_degenerate_faces()

        print("All tests passed!")

    finally:
        # Clean up the temporary directory
        shutil.rmtree(tmpdir)
        print(f"Cleaned up temporary files in {tmpdir}")

if __name__ == "__main__":
    run_tests()