import bpy

import numpy as np
import sys

from model_construct import mesh3D

# Quadric error metrics based error (Garland and Heckbert).
# Mesh simplification calss


class gh_mesh_simplify(mesh3D):
    def __init__(self, threshold, simplification_ratio, obj_filepath):
        if simplification_ratio > 1 or simplification_ratio <= 0:
            sys.exit('Error: simplification ratio should be in (0;1]).')
        if threshold < 0:
            sys.exit('Error: threshold should be non-negative.')
        self.obj_filepath = obj_filepath
        self.t = threshold
        self.ratio = simplification_ratio

        # Function to load an OBJ file into Blender
    def load_obj(filepath):
        bpy.ops.wm.obj_import(filepath=filepath)

    # Function to run the simplification process
    def simplify_mesh():
        active_object = bpy.context.active_object
        if active_object is None or active_object.type != 'MESH':
            raise ValueError("No active mesh object selected in Blender.")

        model = mesh3D(active_object)
        model.load_from_blender()
        model.initialization_simplification()
        # Insert the additional simplification code here
        model.update_blender_mesh()


# # Load your OBJ file (provide the correct path)
# obj_filepath = "D:\\Math-Concepts-For-Devs\\00.Project\\Mesh_simplification_python-master\\models\\dinosaur.obj"
# load_obj(obj_filepath)

# # Run the simplification process
# simplify_mesh()
