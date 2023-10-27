# Step Count and Data Generation Algorithms

This repository contains two scripts developed for step count estimation and data generation, along with supporting code and examples. These algorithms are designed to help in the analysis and processing of acceleration data, especially for applications related to step counting and data synthesis.

## Algorithms

### 1. Step Count Estimation Algorithm
The step count estimation algorithm is implemented in the `git_step_count_developed_algorithm` script. This algorithm provides a method for estimating the number of steps from an acceleration signal. It utilises various signal processing and analysis techniques to identify peaks or step occurrences in the data. 

### 2. Data Generation Algorithm
The data generation algorithm is designed to create synthetic acceleration data for various applications, including testing and prototyping.

## Libraries Used

To implement these algorithms, we make use of several Python libraries:

- numyy: For efficient numerical operations and array manipulations.
- signalz: Provides functionality for generating synthetic signals and noise.
- matplotlib: Used for data visualisation and plotting.
- scipy: Offers tools for signal processing, optimisation, and peak detection.
- tslearn: Used for time series data manipulation and dynamic time warping (DTW) barycenter averaging.
- pandas: Utilised for data manipulation and handling datasets.

## Getting Started

To get started with these algorithms, please refer to the documentation and examples provided in the respective subdirectories of this repository.

- For the step count estimation algorithm, explore the `git_step_count_developed_algorithm` directory.
- For the data generation algorithm, refer to the `git_acc_synthetic_signal_generatior` directory.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to reach out if you have any questions or need assistance with using these algorithms. We hope these algorithms are helpful for your projects and research.
