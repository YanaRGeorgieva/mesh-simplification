import os
import trimesh
import numpy as np
import pymesh
import pandas as pd

def check_manifold_and_orientation(mesh):
    """
    Check if the mesh is vertex-manifold, edge-manifold, and oriented.
    """
    # Check for vertex manifold by ensuring each vertex is part of a well-connected neighborhood
    vertex_manifold = all(mesh.vertex_neighbors)

    # Check for edge manifold by ensuring each edge is shared by at most two faces
    edge_manifold = True
    non_manifold_edges = mesh.edges_nonmanifold
    if len(non_manifold_edges) > 0:
        edge_manifold = False

    # Check if the mesh is oriented (faces have consistent winding)
    oriented = mesh.is_winding_consistent

    return vertex_manifold, edge_manifold, oriented

def extract_mesh_features(mesh_file, file_id):
    # Load the mesh using trimesh
    mesh = trimesh.load(mesh_file)

    # Initialize dictionary to store features
    features = {
        'file_id': file_id,
        'num_vertices': len(mesh.vertices),
        'num_faces': len(mesh.faces),
        'total_area': mesh.area,
        'volume': mesh.volume,
        'bounding_box_dimensions': mesh.bounding_box.extents.tolist(),
        'bounding_box_volume': mesh.bounding_box.volume,
        'euler_characteristic': mesh.euler_number,
        'is_watertight': mesh.is_watertight,
        'is_convex': mesh.is_convex,
        'center_mass': mesh.center_mass.tolist(),
        'moment_inertia': mesh.moment_inertia.tolist(),
        'principal_inertia_components': mesh.principal_inertia_components.tolist(),
        'surface_area_to_volume_ratio': mesh.area / mesh.volume,
        'convex_hull_volume': mesh.convex_hull.volume,
        'convex_hull_area': mesh.convex_hull.area
    }

    # Vertex valence
    valences = np.zeros(len(mesh.vertices))
    for face in mesh.faces:
        for vertex in face:
            valences[vertex] += 1

    features.update({
        'min_valance': np.min(valences),
        'median_valance': np.median(valences),
        'max_valance': np.max(valences),
        'ave_valance': np.mean(valences),
    })

    # Dihedral angles
    dihedral_angles = mesh.face_adjacency_angles

    features.update({
        'min_dihedral_angle': np.min(dihedral_angles),
        'median_dihedral_angle': np.median(dihedral_angles),
        'max_dihedral_angle': np.max(dihedral_angles),
        'ave_dihedral_angle': np.mean(dihedral_angles),
    })

    # Aspect ratios
    aspect_ratios = mesh.face_aspect_ratios

    features.update({
        'min_aspect_ratio': np.min(aspect_ratios),
        'median_aspect_ratio': np.median(aspect_ratios),
        'max_aspect_ratio': np.max(aspect_ratios),
        'ave_aspect_ratio': np.mean(aspect_ratios),
    })

    # Calculate geometrical degeneracies (faces with very small areas)
    min_face_area = 1e-10  # Threshold to consider a face as degenerate
    degenerate_faces = mesh.area_faces < min_face_area
    num_geometrical_degenerated_faces = np.sum(degenerate_faces)

    features.update({
        'num_geometrical_degenerated_faces': num_geometrical_degenerated_faces,
    })

    # Calculate combinatorial degeneracies (faces with repeated vertices)
    combinatorial_degeneracies = 0
    for face in mesh.faces:
        if len(set(face)) < 3:
            combinatorial_degeneracies += 1

    features.update({
        'num_combinatorial_degenerated_faces': combinatorial_degeneracies,
    })

    # Number of connected components
    features.update({
        'num_connected_components': len(mesh.split(only_watertight=False)),
    })

    # Boundary edges (edges that belong to only one face)
    boundary_edges = mesh.edges_boundary
    features.update({
        'num_boundary_edges': len(boundary_edges),
    })

    # Duplicated faces
    unique_faces = set()
    num_duplicated_faces = 0
    for face in mesh.faces:
        face_tuple = tuple(sorted(face))
        if face_tuple in unique_faces:
            num_duplicated_faces += 1
        else:
            unique_faces.add(face_tuple)

    features.update({
        'num_duplicated_faces': num_duplicated_faces,
    })

    # Self-intersections (using PyMesh)
    py_mesh = pymesh.load_mesh(mesh_file)
    num_self_intersections = len(pymesh.detect_self_intersection(py_mesh))

    features.update({
        'num_self_intersections': num_self_intersections,
    })

    # Euler characteristic (calculated by pymesh)
    features.update({
        'euler_characteristic': pymesh.euler_characteristic(py_mesh),
    })

    # Manifold checks
    vertex_manifold, edge_manifold, oriented = check_manifold_and_orientation(mesh)
    features.update({
        'vertex_manifold': vertex_manifold,
        'edge_manifold': edge_manifold,
        'oriented': oriented
    })

    return features

def process_mesh_folder(input_folder, output_csv):
    all_features = []

    for filename in os.listdir(input_folder):
        if filename.endswith(('.obj', '.stl', '.ply')):
            # Extract file ID (assuming the ID is the first part of the filename before an underscore)
            file_id = filename.split('_')[0]
            file_path = os.path.join(input_folder, filename)
            features = extract_mesh_features(file_path, file_id)
            all_features.append(features)

    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)

# Example usage
input_folder = 'path/to/input/folder'
output_csv = 'output_mesh_features.csv'
process_mesh_folder(input_folder, output_csv)
