import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor
from garland_heckbert_algorithm import Mesh3D

def generate_output_filename(input_file, output_dir, threshold, simplification_ratio, penalty_weight):
    # Extract the base filename without extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    # Create the new filename with suffixes
    suffix = f"_t{threshold}_r{simplification_ratio}_p{penalty_weight}"
    # Preserve the original extension
    extension = os.path.splitext(input_file)[1]
    # Combine to form the output filename
    return os.path.join(output_dir, f"{base_name}{suffix}{extension}")

def process_mesh_file(input_file, output_dir, threshold, simplification_ratio, penalty_weight):
    output_file = generate_output_filename(input_file, output_dir, threshold, simplification_ratio, penalty_weight)

    mesh_simplifier = Mesh3D(threshold=threshold, simplification_ratio=simplification_ratio, penalty_weight=penalty_weight)

    mesh_simplifier.simplify_obj_from_file(input_file, output_file)

def main():
    parser = argparse.ArgumentParser(description="Simplify 3D meshes in a folder using Garland and Heckbert's algorithm.")

    parser.add_argument('input_folder', type=str, help="Folder containing the 3D model files (OBJ/PLY/STL).")
    parser.add_argument('output_folder', type=str, help="Output folder where the simplified models will be saved.")

    parser.add_argument('--threshold', type=float, default=0.1, help="Distance threshold for selecting valid pairs (default: 0.1).")
    parser.add_argument('--simplification_ratio', type=float, default=0.5, help="The ratio to simplify the mesh (default: 0.5).")
    parser.add_argument('--penalty_weight', type=float, default=2000.0, help="Penalty weight for boundary or discontinuity edges (default: 2000.0).")

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Gather all mesh files
    supported_formats = ['.obj', '.ply', '.stl']
    mesh_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in supported_formats]

    if not mesh_files:
        print(f"No supported mesh files found in the directory: {input_folder}")
        sys.exit(1)

    # Process each mesh file in parallel
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_mesh_file, mesh_file, output_folder, args.threshold, args.simplification_ratio, args.penalty_weight)
            for mesh_file in mesh_files
        ]
        for future in futures:
            future.result()  # Wait for all tasks to complete

    print(f"Processing complete. Simplified meshes saved to: {output_folder}")

if __name__ == "__main__":
    main()
