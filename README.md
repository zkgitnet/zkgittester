# ZK Git Tester

## Description
The ZK Git Tester is used to perform automated performance evaluations of the [ZK Git Tool](https://github.com/zkgitnet).  It requires a JSON configuration file that defines the repositories and remotes to be used as test data. The configuration also specifies how many data points to generate per remote and repository.

Test results are visualized as bar or box plots in SVG format. Additionally, the raw data for each individual data point is saved for further analysis.

In addition, it contains a short script that can perform statistical power analysis in order to determine sample- or effect size.

## Process
1. **Clone** the repositories to your local machine (e.g., from GitHub’s trending list).
2. **Configure the remotes** for each repository.  
   If `git-crypt` is one of the remotes, duplicate the cloned repository into a new directory with the suffix `_gitcrypt` (e.g., `repo1` → `repo1_gitcrypt`), then initialize and commit the encrypted version.
3. **Create the JSON configuration** using the provided template.  
   This file must include:
   - the path to each repository,
   - the name of the primary branch,
   - and the list of remotes.
4. **Set** the `TEST_NUM_ROUNDS` value in the configuration file to define the desired number of data points.
5. **Start the application**.

## Raw Data
The raw data of the performance evaluation is available on [Zenodo](https://doi.org/10.5281/zenodo.15399286).
