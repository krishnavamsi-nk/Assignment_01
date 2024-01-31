# Assignment_01

# Py-Spy Profiling ReadMe

## Introduction

This repository contains code for a Python script along with instructions on how to use `py-spy` for profiling. `py-spy` is a sampling profiler for Python programs.

## Installation

1. Install `py-spy` using pip:

    ```
    pip install py-spy
    ```

## Usage

1. Run your Python script with `py-spy` for profiling:

    ```
    py-spy top -- python your_script.py
    example:
        py-spy top -- python Wine_Quality_Classification_using_ANN.py
    ```

    This command will start profiling your Python script and display real-time profiling information.

2. Press `Ctrl+C` to stop the profiling.

3. View the generated flame graph in your terminal or export it to a file:

    ```
    py-spy top -- python your_script.py --flame profile.svg

    example :
         py-spy top -- python Wine_Quality_Classification_using_ANN.py --flame profile.svg
    ```

    Replace `profile.svg` with your desired file name.

## Additional Notes

- For more advanced usage and options, refer to the `py-spy` documentation: [https://github.com/benfred/py-spy](https://github.com/benfred/py-spy)

- Make sure to have the necessary permissions to install packages and run the profiling.

- This guide assumes you have Python and pip already installed on your system.

