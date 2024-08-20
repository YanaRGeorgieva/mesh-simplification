import trimesh
import numpy as np
from scipy.spatial import cKDTree
import os
import csv
import argparse

def hausdorff_distance(mesh1, mesh2):
    """
    Calculate the Hausdorff Distance between two meshes.
    """
    # Sample points on the surface of each mesh
    points1 = mesh1.sample(10000)
    points2 = mesh2.sample(10000)
    
    # Create KD-trees for fast nearest neighbor search
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    # Compute distances from each point in mesh1 to the nearest point in mesh2
    distances_1_to_2, _ = tree2.query(points1, k=1)
    
    # Compute distances from each point in mesh2 to the nearest point in mesh1
    distances_2_to_1, _ = tree1.query(points2, k=1)
    
    # The Hausdorff distance is the maximum of these distances
    hausdorff_dist = max(distances_1_to_2.max(), distances_2_to_1.max())
    return hausdorff_dist

def rmse(mesh1, mesh2):
    """
    Calculate the RMSE between two meshes.
    This measures how well the simplified mesh (mesh2) approximates the original mesh (mesh1).
    """
    # Sample points on the surface of each mesh
    points1 = mesh1.sample(10000)
    points2 = mesh2.sample(10000)
    
    # Create a KD-tree for the original mesh points
    tree1 = cKDTree(points1)
    
    # Compute distances from each point in the simplified mesh (mesh2) to the nearest point in the original mesh (mesh1)
    distances, _ = tree1.query(points2, k=1)
    
    # Compute RMSE
    rmse_value = np.sqrt(np.mean(distances**2))
    return rmse_value

def process_meshes(original_folder, simplified_folder, output_csv):
    # Initialize lists to hold results
    file_ids = []
    hausdorff_distances = []
    rmses = []

    # List all files in the original folder
    original_files = {os.path.splitext(f)[0]: f for f in os.listdir(original_folder) if f.endswith('.obj')}
    
    # Iterate over original files
    for prefix, original_filename in original_files.items():
        original_filepath = os.path.join(original_folder, original_filename)

        # Find the corresponding simplified files with the same prefix
        simplified_files = [f for f in os.listdir(simplified_folder) if f.startswith(prefix) and f.endswith('.obj')]

        for simplified_filename in simplified_files:
            simplified_filepath = os.path.join(simplified_folder, simplified_filename)

            # Load the meshes
            original_mesh = trimesh.load(original_filepath)
            simplified_mesh = trimesh.load(simplified_filepath)

            # Calculate Hausdorff Distance
            hausdorff_dist = hausdorff_distance(original_mesh, simplified_mesh)

            # Calculate RMSE
            rmse_value = rmse(original_mesh, simplified_mesh)

            # Append results to the lists
            file_ids.append(prefix)
            hausdorff_distances.append(hausdorff_dist)
            rmses.append(rmse_value)

    # Save the results to a CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(['file_id', 'hausdorff_distance', 'rmse'])
        # Write the data
        for file_id, hausdorff_dist, rmse_value in zip(file_ids, hausdorff_distances, rmses):
            csvwriter.writerow([file_id, hausdorff_dist, rmse_value])

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Calculate Hausdorff Distance and RMSE between original and simplified meshes.')
    parser.add_argument('original_folder', type=str, help='Path to the folder containing original meshes.')
    parser.add_argument('simplified_folder', type=str, help='Path to the folder containing simplified meshes.')
    parser.add_argument('output_csv', type=str, help='Path to the output CSV file.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Process the meshes and save the results
    process_meshes(args.original_folder, args.simplified_folder, args.output_csv)

if __name__ == "__main__":
    main()