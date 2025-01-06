# Advanced Parallel Programming

**Project:** Template Visual Studio Code Project Config (CUDA, MPI, OpenMP)

**Author:** Mohsen Gholami | iMohsen02  
**University:** Ferdowsi University of Mashhad (FUM)

---

## Overview

This project serves as a template for implementing parallel programming solutions using CUDA, MPI, and OpenMP. The setup is designed for easy integration with VS Code and supports local and network execution.

---

## Project Structure

```
.
├── README.md
├── .vscode
│   ├── c_cpp_properties.json
│   ├── settings.json
│   └── tasks.json
├── bin
│   ├── main.exe
│   ├── main.exp
│   └── main.lib
├── src
│   ├── main.cu
│   ├── algorithm
│   ├── models
│   └── utils
└── test
    └── test.cu
```

### Directory Details

-   **.vscode/**: Contains VS Code configuration files for building and running the project.
-   **bin/**: Stores compiled binaries and executable files.
-   **src/**: Source code directory where all `.cu` files and related modules reside.
    -   `main.cu`: Entry point for the project.
    -   `algorithm/`: Directory for algorithms.
    -   `models/`: Directory for data models.
    -   `utils/`: Directory for utility functions.
-   **test/**: Contains test files for validating the implementation.

---

## How to Use

before start make sure CUDA, CL(visual studio compiler) and MPI be be installed.

### 1. Setting Up the Project

1. Copy the `sample-vscode-config` folder and rename it to your project name.
2. Replace `D:/MPI/SDK/Include` and `D:/MPI/SDK/Lib/x64` with your system paths in `.vscode/tasks.json`.
3. Write your code in `src/main.cu` or create additional files in the appropriate subdirectories.

---

### 2. Building and Running the Project Using VS Code Tasks

This project includes pre-configured tasks in the `tasks.json` file for building and running your program. Follow these steps:

#### Build the Program

1. Open the project in VS Code.
2. Build the program by running the following task:
    ```
    Top Menu -> Terminal -> Run Task -> Build-64 Program (CUDA, MPI, OPENMP)
    ```
    - This compiles the current open file in the editor using NVCC with support for MPI and OpenMP.
    - The compiled binary is saved in the `bin/` directory.

#### Run the Program Locally

1. Run the program locally using the following task:
    ```
    Top Menu -> Terminal -> Run Task -> Run Program (CUDA, MPI, OPENMP)
    ```
2. Enter the required inputs when prompted:
    - **Number of MPI processes**: Specify the number of processes (default is `2`).
    - **argv-1**: First command-line argument for the program (default: `No argv[1]`).
    - **argv-2**: Second command-line argument for the program (default: `No argv[2]`).

#### Clean the Bin Directory

-   To remove previously compiled files from the `bin/` directory, run the following task:
    ```
    Top Menu -> Terminal -> Run Task -> Clean Bin Directory
    ```

#### Build and Run Sequentially

-   To clean, build, and run the program in sequence, execute:
    ```
    Top Menu -> Terminal -> Run Task -> Build and Run
    ```
    -   This task performs all the steps (clean, build, and run) in the correct order.

---

### 3. Running the Program on a Network

1. Create a directory like example in the shared path:
    ```
    \\ENG-LAB306-C9\mpi\advance-parallel-programming-cuda-projects\
    ```
2. Place all files from the `bin/` directory into your folder.
3. Run the program using the following command:
    ```bash
    mpiexec -hosts 2 172.21.50.42 172.21.51.186 \\\\ENG-LAB306-C9\\mpi\\advance-parallel-programming-cuda-projects\\{your-folder}\\{your-exe-file}.exe
    ```
    Replace `{your-folder}` and `{your-exe-file}` with your folder and executable file names(shared path, ips are configured in FUM Lab, if you are going to use it anywhere update it).

---

### 4. Notes

-   When using VS Code tasks, ensure the target file is open before executing a task.
-   Update paths for MPI include and library directories in `tasks.json` if necessary.
    -   Replace `D:/MPI/SDK/Include` and `D:/MPI/SDK/Lib/x64` with your system paths.
-   The VS Code configuration files (`tasks.json`, `c_cpp_properties.json`, `settings.json`)
-   Make sure parent folder name and sub-folders name of project contains no white-space

are pre-configured for this project. Adjust them if necessary.

---

## Contact

For any questions or feedback, contact **Mohsen Gholami** at **[iMohsen02](https://t.me/iMohsen02)**.
