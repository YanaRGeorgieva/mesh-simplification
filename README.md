# Module 1: Math for Developers project
**Project Focus:** Implementing a basic mesh simplification algorithm

**Project Description:**
- **Algorithm Implementation:** Implement the [Garland and Heckbert (1997)](https://www.cs.cmu.edu/~garland/Papers/quadrics.pdf) mesh simplification algorithm in Python. If any time is left do some improvements with the error metrics proposed in [Elena Ovreiu. Accurate 3D mesh simplification (2012)](https://theses.hal.science/tel-01224848/file/these.pdf).
- **Blender Integration:** Use the Blender Python API to integrate the algorithm. The script should take an input mesh, apply the simplification algorithm, and output a simplified mesh.
- **Mathematical Concepts:** Emphasize the mathematical foundations such as matrices, vectors, and possibly complex numbers if needed for certain transformations or calculations.
- **Visualization:** Use Blender's scripting view to visualize the before and after of the mesh simplification.

**Deliverables:**
- Python script for the mesh simplification algorithm.
- Documentation explaining the mathematical concepts used.
- Demonstration of the script running within Blender.

# Module 2: Data Science
**Project Focus:** Analyzing existing 3D mesh data with the implemented simplification algorithm

**Project Description:**
- **Use Pre-existing Datasets:** Utilize publicly available 3D mesh datasets from online sources. I will use [Thingi10K](https://ten-thousand-models.appspot.com/).
- **Feature Extraction:** Focus on extracting relevant features from these meshes before (some are extracted by the [authors of teh dataset](https://docs.google.com/spreadsheets/d/1ZM5_1ry3Oe5uDJZxQIcFR6fjjas5rX4yjkhQ8p7Kf2Q/edit?usp=sharing)) and after applying your simplification algorithm. Features can include vertex count, triangle count, surface area, etc.
- **Data Analysis:** Perform basic data analysis to assess the impact of the simplification algorithm on these features. Use statistical methods to quantify changes and assess algorithm effectiveness.
- **Exploratory Data Analysis (EDA):** Create visualizations to illustrate the impact of the simplification algorithm on various types of meshes. I will use tools like Matplotlib or Seaborn for plotting.
- **Documentation and Reporting:** Document your analysis process and findings, focusing on how the algorithm performs across different datasets and highlighting any interesting insights or trends.

**Deliverables:**
- Scripts for applying the simplification algorithm to pre-existing datasets.
- Data analysis scripts for feature extraction and visualization.
- Visualizations showing the impact of simplification on various mesh features.
- A report summarizing the analysis, including visualizations and key insights.

## Requirements
In the requirements.txt file.

## How to run it
1. You can directly put this code in Blender scripting and run it on a preselected active mesh.
2. Otherwise, that is why we need fake-bpy-module installed and Blender Development extension set up on VSCode.

    * You have to first start a debug Blender instance via the command palette "Blender: Start".
    * VSCode will set up connect to it in debug mode.
    * You can load some meshes and be sure to select one of them to be the active object.
    * Then you run the script via the command palette "Blender: Run Script"

3. There is a third option involving the `load_file` and `output_file` methods for files (supports whatever the [trimesh](https://trimesh.org/index.html#) library supports, but in general it works with *.obj, *.stl and *.ply files). In this case you can run the program in a terminal (I am sorry but for now you have to hardcode the paths to the files).

4. A fourth option using the CLI scripts found in the .src folder.

## TODOs:
 - Optimize code more;
 - More examples and better documentation;

<img src="./data/old_data/images/trexFull.png" alt="Tyrannosaurus no simplification" width="45%"> <img src="./data/old_data/images/trex90P.png" alt="Tyrannosaurus 90% simplification" width="45%">

