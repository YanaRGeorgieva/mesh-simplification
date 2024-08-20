import argparse
from garland_heckbert_algorithm import Mesh3D

def main():
    parser = argparse.ArgumentParser(
        description="Simplify 3D meshes using Garland and Heckbert's algorithm with boundary preservation."
    )

    # Input and output files
    parser.add_argument(
        "input_file",
        type=str,
        help="The input 3D model file."
    )
    
    parser.add_argument(
        "output_file",
        type=str,
        help="The output file to save the simplified 3D model."
    )

    # Simplification ratio
    parser.add_argument(
        "--simplification_ratio",
        type=float,
        default=0.5,
        help="The ratio to simplify the mesh. Value should be between 0 and 1 (default: 0.5)."
    )

    # Penalty weight for boundary edges
    parser.add_argument(
        "--penalty_weight",
        type=float,
        default=1000.0,
        help="The penalty weight for boundary or discontinuity edges (default: 1000.0)."
    )

    # Distance threshold for valid pairs
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Distance threshold for selecting valid pairs (default: 0.1)."
    )

    args = parser.parse_args()

    # Create the mesh simplifier with provided arguments
    meshy = Mesh3D(threshold=args.threshold, simplification_ratio=args.simplification_ratio, penalty_weight=args.penalty_weight)

    # Load, simplify, and save the mesh
    meshy.simplify_obj_from_file(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
