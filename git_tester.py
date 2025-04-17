"""
This module contains functions for performing performance measurements
and statistics on Git operations, such as memory and CPU usage,
network, and disk I/O during Git pushes to different remotes. It
provides functionality for logging, generating plots, and calculating
statistics on performance data.

Main Functionalities:
1. **Performance Measurement**: Includes decorators (`measure_performance`)
to measure execution time, network I/O, disk I/O, memory usage
and CPU usage of functions such as `git_push`.
2. **Process Management**: Functions like `find_processes` and
`run_command` are used to manage subprocesses and interact with the system.
3. **Data Collection and Storage**: The module collects data on system
resources (memory, CPU, network, disk) during Git operations and
stores them in `performance_data`.
4. **Git Operations**: Functions like `git_push` and `delete_remote_repo`
handle Git push and remote deletion, including using `git-crypt` for encryption.
5. **Data Visualization**: Generates boxplots to visualize performance
metrics across different repositories and remotes using `generate_box_plots`.
6. **Statistics Calculation**: Calculates statistics (average and
standard deviation) for the collected performance data and writes
them to a file using `calculate_statistics`.

Usage:
1. The module reads repository information from a JSON file using `read_repository_information`.
2. It runs Git operations (`git_push`) for each repository and remote
across multiple rounds, while monitoring memory and CPU usage in separate threads.
3. After all operations are completed, it generates boxplots and calculates
performance statistics for each repository and remote.

"""

import time
import os
import re
import json
import shutil
import subprocess
import statistics
import threading
import itertools
from datetime import datetime
from functools import wraps
from scipy import stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import config
import logger

performance_data = {}
LOGGER = logger.configure_logger()

def load_existing_data():
    """Loads performance data from JSON files in the ./tmp directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(script_dir, config.DIR_TMP)

    for filename in os.listdir(tmp_dir):
        if not filename.endswith(config.FILE_JSON):
            continue

        path = os.path.join(tmp_dir, filename)

        with open(path, config.FILE_READ, encoding=config.FILE_UTF8) as f:
            entry = json.load(f)

        if isinstance(entry, dict) and all(isinstance(v, dict) for v in entry.values()):
            for repo, remotes in entry.items():
                for remote, entries in remotes.items():
                    if repo not in performance_data:
                        performance_data[repo] = {}
                    if remote not in performance_data[repo]:
                        performance_data[repo][remote] = []
                    performance_data[repo][remote].extend(entries)
        else:
            match = re.match(r"(.+?)_(.+?)_round", filename)
            if match:
                repo, remote = match.groups()
            else:
                repo, remote = "unknown_repo", "unknown_remote"

            if repo not in performance_data:
                performance_data[repo] = {}
            if remote not in performance_data[repo]:
                performance_data[repo][remote] = []

            performance_data[repo][remote].append(entry)


def get_round_data_path(repo_dir, remote, round_entry):
    """Returns the path to save a round's data in ./tmp, relative to the script directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(script_dir, config.DIR_TMP)
    os.makedirs(tmp_dir, exist_ok=True)

    round_number = round_entry[config.STATS_ROUND]

    filename = f"{os.path.basename(repo_dir)}_{remote}_round_{round_number}.json"

    return os.path.join(tmp_dir, filename)


def save_round_data(repo_dir, remote, round_entry):
    """Saves a single round's performance data to ./tmp directory."""
    path = get_round_data_path(repo_dir, remote, round_entry)

    with open(path, config.FILE_WRITE, encoding=config.FILE_UTF8) as f:
        json.dump(round_entry, f, indent=4)


def generate_box_plots(data):
    """Generates boxplots for each metric, grouped by repo_dir and remote."""
    if not data:
        LOGGER.warning(config.ERROR_NO_BOXPLOT)
        return

    records = []
    for repo_dir, remotes in data.items():
        for remote, entries in remotes.items():
            for entry in entries:
                for metric in config.STATS:
                    if metric in entry:
                        records.append({
                            config.STATS_REPODIR: os.path.basename(repo_dir),
                            config.STATS_REMOTE: remote,
                            config.STATS_METRIC: metric,
                            config.STATS_VALUE: entry[metric]
                        })

    df = pd.DataFrame(records)

    if df.empty:
        LOGGER.warning(config.ERROR_NO_METRICS)
        return

    for metric in config.STATS:
        metric_df = df[df[config.STATS_METRIC] == metric]
        if metric_df.empty:
            continue

        plt.figure(figsize=(6, 6))
        sns.boxplot(data=metric_df, x=config.STATS_REPODIR,
                    y=config.STATS_VALUE, hue=config.STATS_REMOTE)
        
        x_order = metric_df[config.STATS_REPODIR].unique()
        hue_order = metric_df[config.STATS_REMOTE].unique()
        
        # Total number of hue levels (remotes) per repo
        num_hue = len(hue_order)
        ax = plt.gca()

        # Get hatch styles and create a hatch map for consistent styling
        hatch_styles = ['///', '\\\\\\', 'xxx', '---']
        unique_remotes = metric_df[config.STATS_REMOTE].unique()
        hatch_map = {remote: hatch_styles[i % len(hatch_styles)] for i, remote in enumerate(unique_remotes)}

        # Extract the legend mapping (label to artist)
        handles, labels = ax.get_legend_handles_labels()

        # Create reverse legend map to match labels to remotes
        label_to_remote = {v: k for k, v in {
            "gitremote": "git",
            "gitcryptremote": "git-grypt",
            "gcryptremote": "gcrypt",
            "zkgitremote": "ZK Git"
        }.items()}

        # Fix inconsistent hatch styles by examining each patch’s color (which encodes the hue)
        # Create a mapping from facecolor to hatch
        color_to_remote = {
            handle.get_facecolor(): label for handle, label in zip(handles, labels)
        }
        remote_to_hatch = {
            label: hatch_map.get(label, '') for label in labels
        }

        # Apply hatch pattern based on the color (which maps to remote)
        for patch in ax.patches:
            facecolor = patch.get_facecolor()
            remote_label = color_to_remote.get(facecolor)
            if remote_label:
                hatch = hatch_map.get(remote_label, '')
                patch.set_facecolor('white') 
                patch.set_hatch(hatch)
                patch.set_edgecolor('black')
        

        handles, labels = plt.gca().get_legend_handles_labels()
        
        label_map = {
            "gitremote": "git",
            "gitcryptremote": "git-grypt",
            "gcryptremote": "gcrypt",
            "zkgitremote": "ZK Git"
        }
        new_labels = [label_map.get(label, label) for label in labels]

        plt.legend(handles=handles, labels=new_labels, title=None)
        plt.title("")
        plt.xlabel("")
        plt.ylabel(metric.replace('_', ' ').title())
        if metric == config.STATS_AVG_CPU:
            plt.ylim(bottom=0, top=200)
        else:
            y_max = metric_df[config.STATS_VALUE].max()
            plt.ylim(bottom=0, top=y_max * 1.1)
        plt.grid(True)
        plt.xticks(rotation=90)
        plt.tight_layout()

        timestamp = datetime.now().strftime(config.DATETIME_FORMAT)
        filename = f"outputs/{timestamp}_{metric}_comparison.png"
        plt.savefig(filename)
        LOGGER.info(config.LOG_BOXPLOT, filename)
        plt.show()

        # Concentated boxplots
        remote_palette = {
            "gitremote": "#1f77b4",       # blue
            "gitcryptremote": "#ff7f0e",  # orange
            "gcryptremote": "#2ca02c",    # green
            "zkgitremote": "#d62728"      # red
        }
        plt.figure(figsize=(2, 6))
        sns.boxplot(data=metric_df, hue=config.STATS_REMOTE,
                    y=config.STATS_VALUE, legend=False)

        handles, labels = plt.gca().get_legend_handles_labels()
        label_map = {
            "gitremote": "git",
            "gitcryptremote": "git-crypt",
            "gcryptremote": "gcrypt",
            "zkgitremote": "ZK Git"
        }
        new_xticklabels = [label_map.get(t.get_text(), t.get_text()) for t in plt.gca().get_xticklabels()]
        plt.gca().set_xticklabels(new_xticklabels)

        plt.title("")
        plt.xlabel("")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.ylim(bottom=0)
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.tight_layout()

        timestamp = datetime.now().strftime(config.DATETIME_FORMAT)
        filename = f"outputs/{timestamp}_{metric}_combined_comparison.png"
        plt.savefig(filename)
        LOGGER.info(config.LOG_BOXPLOT, filename)
        plt.show()


def find_processes(terms):
    """Finds all process PIDs where the command contains any
    of the given search strings from the list of terms."""
    matching_pids = []
    for process in psutil.process_iter(attrs=[config.PID, config.CMDLINE]):
        try:
            cmdline = process.info[config.CMDLINE]
            if cmdline and any(term in cmd for term in terms for cmd in cmdline):
                matching_pids.append(process.info[config.PID])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            LOGGER.warning(config.ERROR_NO_PID)
            continue

    return matching_pids


def measure_performance(func):
    """Decorator to measure execution time, memory usage, and I/O stats of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        net_before = psutil.net_io_counters()
        disk_before = psutil.disk_io_counters()

        result = func(*args, **kwargs)

        execution_time = time.time() - start_time
        net_after = psutil.net_io_counters()
        disk_after = psutil.disk_io_counters()

        net_sent = (net_after.bytes_sent - net_before.bytes_sent) / config.STATS_MB
        net_recv = (net_after.bytes_recv - net_before.bytes_recv) / config.STATS_MB
        disk_read = (disk_after.read_bytes - disk_before.read_bytes) / config.STATS_MB
        disk_write = (disk_after.write_bytes - disk_before.write_bytes) / config.STATS_MB

        repo_dir = args[0]
        remote = args[1]
        round_num = args[3]

        existing_round_entry = next((entry for entry
                                     in performance_data[os.path.basename(repo_dir)][remote]
                                     if entry.get(config.STATS_ROUND) == round_num), None)

        if existing_round_entry is not None:
            existing_round_entry.update({
                config.STATS_EXECUTION_TIME: execution_time,
                config.STATS_NET_SENT: net_sent,
                config.STATS_NET_RECEIVED: net_recv,
                config.STATS_DISK_READ: disk_read,
                config.STATS_DISK_WRITE: disk_write
            })

        save_round_data(repo_dir, remote, existing_round_entry)

        LOGGER.debug(config.LOG_ROUND_DATA, os.path.basename(repo_dir), remote, round_num + 1)
        LOGGER.debug(config.LOG_ROUND_TIME, execution_time)
        LOGGER.debug(config.LOG_ROUND_NET, net_sent, net_recv)
        LOGGER.debug(config.LOG_ROUND_DISK, disk_read, disk_write)
        LOGGER.debug(config.LOG_ROUND_MEM, existing_round_entry.get(config.STATS_PEAK_MEMORY))
        LOGGER.debug(config.LOG_ROUND_CPU, existing_round_entry.get(config.STATS_AVG_CPU))

        return result

    return wrapper


def read_repository_information(json_file):
    """Reads the repository information from a JSON file"""
    with open(json_file, config.FILE_READ, encoding=config.FILE_UTF8) as input_file:
        data = json.load(input_file)
    return data[config.JSON_REPOS]


def delete_remote_branch(directory, remote, branch):
    """Deletes all the remote content from the specified directory."""
    if not os.path.isdir(directory):
        LOGGER.error(config.LOG_DIRECTORY, directory)
        return

    try:
        os.chdir(directory)
        process = subprocess.Popen(
            [config.GIT_GIT, config.GIT_PUSH, config.GIT_DELETE, remote, branch],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        LOGGER.debug(config.LOG_RESET_REPO, process.pid)
        process.wait()
    except subprocess.CalledProcessError as e:
        LOGGER.warning(config.LOG_DELETE_FAIL, e.stderr)
    except Exception as e:
        LOGGER.critical(config.LOG_ERROR, e)


def create_and_commit_file(repo_dir, remote, filename="random_file.bin", commit_msg="Add 1MB random file"):
    """Creates a 1MB random file, adds it to the Git repo, and commits it."""
    if remote.lower() == config.GIT_CRYPT_REMOTE:
        repo_dir += "_gitcrypt"

    filepath = os.path.join(repo_dir, filename)

    try:
        os.chdir(repo_dir)

        with open(filepath, config.FILE_WRITE_BINARY) as f:
            f.write(os.urandom(1024))

        subprocess.run([config.GIT_GIT, config.GIT_ADD, "*"], check=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)

        subprocess.run([config.GIT_GIT, config.GIT_COMMIT,
                        config.GIT_MESSAGE, commit_msg], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)

        LOGGER.info(config.LOG_COMMIT, filename)

    except subprocess.CalledProcessError as e:
        LOGGER.warning(config.LOG_COMMIT_FAIL, e)
    except Exception as e:
        LOGGER.critical(config.LOG_ERROR, e)


def run_command(command, cwd=None):
    """Runs a subprocess command and waits for it to finish."""
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd
        )
        LOGGER.info(config.LOG_RUN_COMMAND, ' '.join(command), process.pid)
        return process
    except Exception as e:
        LOGGER.critical(config.LOG_ERROR, e)
        raise


def get_memory_usage(pid_list):
    """Returns the total memory usage (in MB) for all given PIDs,
    including their child processes."""

    total_memory = 0.0
    for pid in pid_list:
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            all_processes = [parent] + children

            total_memory += sum(p.memory_info().rss
                                for p in all_processes if p.is_running()) / config.STATS_MB
        except psutil.NoSuchProcess:
            continue

    return total_memory


def get_cpu_usage(pid_list):
    """Returns the total CPU usage (%) for all given PIDs including their child processes."""
    total_cpu = 0.0
    rounds = 0
    for pid in pid_list:
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            all_processes = [parent] + children

            for proc in all_processes:
                if proc.is_running():
                    curr_cpu = proc.cpu_percent(interval=0.1)
                    if curr_cpu > 0:
                        total_cpu += curr_cpu
                        rounds += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return total_cpu, rounds


def monitor_memory(pids, max_memory_usage, process):
    """Monitors the memory usage in a separate thread."""
    while process.poll() is None:
        max_memory_usage[0] = max(max_memory_usage[0], get_memory_usage(pids))
    return max_memory_usage[0]
 
 
def monitor_cpu(pids, tot_cpu_usage, rounds, process):
    """Monitors the CPU usage in a separate thread."""
    while process.poll() is None:
        cpu, r = get_cpu_usage(pids)
        tot_cpu_usage[0] += cpu
        rounds[0] += r
    return tot_cpu_usage[0], rounds[0]


def monitor_process(process, label=config.LABEL_PROCESS):
    """Monitors memory and CPU usage concurrently of the given process until it finishes."""
    max_memory_usage = [0.0]
    tot_cpu_usage = [0.0]
    rounds = [0]
    output = True

    pids = find_processes(config.PID_TERMS)
    if output:
        LOGGER.debug(config.LOG_PROCESS_FOUND, label, pids)
        output = False

    memory_thread = threading.Thread(target=monitor_memory, args=(pids, max_memory_usage, process))
    memory_thread.start()

    cpu_thread = threading.Thread(target=monitor_cpu, args=(pids, tot_cpu_usage, rounds, process))
    cpu_thread.start()

    memory_thread.join()
    cpu_thread.join()

    process.wait()

    return max_memory_usage[0], tot_cpu_usage[0] / rounds[0] if rounds[0] > 0 else 0
 
          
@measure_performance
def git_push(repo_dir, remote, branch, i):
    """Performs a git push in the specified directory."""
    if not os.path.isdir(repo_dir):
        LOGGER.error(config.LOG_DIRECTORY, repo_dir)
        return

    try:
        if remote.lower() == config.GIT_CRYPT_REMOTE:
            os.chdir(repo_dir + "_gitcrypt")            
        else:
            os.chdir(repo_dir)

        max_memory_usage = 0.0
        avg_cpu_usage = 0.0

        if remote.lower() == config.GIT_CRYPT_REMOTE:
            process = run_command([config.GIT_CRYPT, config.GIT_LOCK])
            mem, cpu = monitor_process(process, label=config.LABEL_ENCRYPT)
            max_memory_usage = max(max_memory_usage, mem)
            avg_cpu_usage = max(avg_cpu_usage, cpu)
            process.wait()

        process = run_command([config.GIT_GIT, config.GIT_PUSH, remote, branch, config.GIT_FORCE])
        mem, cpu = monitor_process(process, label=config.LABEL_PUSH)
        max_memory_usage = max(max_memory_usage, mem)
        avg_cpu_usage = max(avg_cpu_usage, cpu)
        process.wait()

        if os.path.basename(repo_dir) not in performance_data:
            performance_data[os.path.basename(repo_dir)] = {}

        if remote not in performance_data[os.path.basename(repo_dir)]:
            performance_data[os.path.basename(repo_dir)][remote] = []

        round_entry = {
            config.STATS_ROUND: i,
            config.STATS_PEAK_MEMORY: max_memory_usage,
            config.STATS_AVG_CPU: avg_cpu_usage
        }

        performance_data[os.path.basename(repo_dir)][remote].append(round_entry)

    except subprocess.CalledProcessError as e:
        LOGGER.error(config.LOG_PUSH_FAIL, e.stderr)
    except Exception as e:
        LOGGER.error(config.LOG_ERROR, e)


def calculate_statistics(data):
    """Calculates statistics (average & standard deviation) from
    the collected performance data and writes to a file. 
    Performs ANOVA and T-tests to compare remotes."""
    if not data:
        LOGGER.warning(config.ERROR_NO_STATS)
        return

    timestamp = datetime.now().strftime(config.DATETIME_FORMAT)
    filename = f"outputs/{timestamp}_stats.txt"

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open(filename, config.FILE_WRITE, encoding=config.FILE_UTF8) as file:

        # === New: collect global (cross-repo) metric values per remote ===
        global_remote_data = {metric: {} for metric in config.STATS}
        for repo_dir, remotes in data.items():
            for remote, entries in remotes.items():
                for metric in config.STATS:
                    values = [entry[metric] for entry in entries if metric in entry]
                    if values:
                        global_remote_data[metric].setdefault(remote, []).extend(values)
        
        for repo_dir, remotes in data.items():
            LOGGER.info(config.LOG_STATS_REPO, os.path.basename(repo_dir))
            file.write(f"\nStatistics for Repository: {os.path.basename(repo_dir)}\n")
            calc_stats = True

            # Perform ANOVA and T-tests within each repository
            for remote, entries in remotes.items():
                num_runs = len(entries)

                if num_runs < 2:
                    LOGGER.warning("\n  Remote: %s (Total Runs: %d)"
                                   + "- Not enough data for standard deviation",
                                   remote, num_runs)
                    file.write(f"\n  Remote: {remote} (Total Runs: "
                               + "{num_runs}) - Not enough data for standard deviation\n")
                    continue

                metric_values = {metric: [entry[metric] for entry in entries
                                          if metric in entry] for metric in config.STATS}

                avg_values = {metric: sum(values) / num_runs
                              for metric, values in metric_values.items()}
                std_values = {
                    metric: statistics.stdev(values) if len(values) > 1 else 0
                    for metric, values in metric_values.items()
                }

                output = f"\n  Remote: {remote} (Total Runs: {num_runs})\n"
                for metric in config.STATS:
                    unit = "sec" if metric == config.STATS_EXECUTION_TIME else "MB"
                    output += (
                        f"  Avg {metric.replace('_', ' ').title()}: {avg_values[metric]:.2f} "
                        f"{unit}, Std Dev: {std_values[metric]:.2f} {unit}\n"
                    )

                # Perform ANOVA if multiple remotes exist in the repository
                if len(remotes) > 1 and calc_stats:
                    remote_data = {metric: [] for metric in config.STATS}

                    for remote_name, remote_entries in remotes.items():
                        for metric in config.STATS:
                            if metric in remote_entries[0]:
                                values = [entry[metric] for entry
                                          in remote_entries if metric in entry]
                                remote_data[metric].append(values)

                    for metric in config.STATS:
                        if len(remote_data[metric]) < 2:
                            LOGGER.warning(config.LOG_METRIC_MISSING, metric,
                                           os.path.basename(repo_dir))
                            continue

                    # Perform ANOVA only on metrics with data for all remotes
                    anova_result = {}
                    for metric, values in remote_data.items():
                        if values:
                            anova_result[metric] = stats.f_oneway(*values)

                    for metric, result in anova_result.items():
                        file.write(f"\nANOVA Result for {metric}: F-statistic = "
                                   f"{result.statistic:.2f}, p-value = {result.pvalue:.8f}\n")
                        LOGGER.info(config.LOG_ANOVA, metric, result.statistic, result.pvalue)

                # Perform pairwise T-tests for each possible pair of remotes
                if len(remotes) > 1 and calc_stats:
                    calc_stats = False
                    for remote1, remote2 in itertools.combinations(remotes.keys(), 2):
                        ttest_result = {}
                        for metric in config.STATS:
                            values1 = [entry[metric] for entry in remotes[remote1] if metric in entry]
                            values2 = [entry[metric] for entry in remotes[remote2] if metric in entry]

                            if values1 and values2:
                                t_stat, p_val = stats.ttest_ind(values1, values2)
                                ttest_result[metric] = {"t-statistic": t_stat, "p-value": p_val}
                                file.write(f"\nT-test Result for {remote1} vs {remote2} ({metric}): "
                                           f"t-statistic = {t_stat:.2f}, p-value = {p_val:.8f}\n")
                                LOGGER.info(config.LOG_TTEST, remote1, remote2, metric, t_stat, p_val)
                            else:
                                file.write(f"\nT-test skipped for {remote1} vs "
                                           f"{remote2} ({metric}) due to missing data.\n")
                                LOGGER.warning(config.LOG_TTEST_FAIL, remote1, remote2, metric)

                LOGGER.info(output)
                file.write(output)

                file.write("\n\n=== Global (Combined) Statistics Across All Repositories ===\n")

        # ANOVA across remotes globally
        file.write("\n--- Global ANOVA Results ---\n")
        for metric, remote_values in global_remote_data.items():
            if len(remote_values) < 2:
                file.write(f"{metric}: Not enough remotes for ANOVA.\n")
                continue
            anova_input = list(remote_values.values())
            result = stats.f_oneway(*anova_input)
            file.write(f"ANOVA for {metric}: F = {result.statistic:.2f}, p = {result.pvalue:.8f}\n")
            LOGGER.info(f"ANOVA for {metric}: F = {result.statistic:.2f}, p = {result.pvalue:.8f}\n")

        file.write("\n--- Global Pairwise T-Tests ---\n")
        for metric, remote_values in global_remote_data.items():
            remotes = list(remote_values.keys())
            for r1, r2 in itertools.combinations(remotes, 2):
                values1 = remote_values[r1]
                values2 = remote_values[r2]
                if values1 and values2:
                    t_stat, p_val = stats.ttest_ind(values1, values2)
                    file.write(f"{metric}: {r1} vs {r2}: t = {t_stat:.2f}, p = {p_val:.8f}\n")
                    LOGGER.info(f"{metric}: {r1} vs {r2}: t = {t_stat:.2f}, p = {p_val:.8f}\n")
                else:
                    file.write(f"{metric}: {r1} vs {r2}: Skipped due to missing data.\n")

        file.write(config.LOG_RAWDATA)
        json.dump(data, file, indent=4)

    LOGGER.debug(config.LOG_STATS_WRITE, filename)


def delete_tmp_directory_contents():
    """Deletes all files inside the ./tmp directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(script_dir, config.DIR_TMP)

    if os.path.exists(tmp_dir):
        for filename in os.listdir(tmp_dir):
            file_path = os.path.join(tmp_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(config.LOG_FILE_DELETE_FAIL, file_path, e)


def main():
    """Main method to conduct all measurements over all repos and remotes"""
    repo_data = read_repository_information(config.JSON_REPOSITORIES)
    total_rounds = sum(len(repo[config.JSON_REMOTES])
                       for repo in repo_data) * config.TEST_NUM_ROUNDS
    load_existing_data()

    current_round = 1
    for repo in repo_data:
        repo_dir = repo[config.JSON_REPODIR]
        branch = repo[config.JSON_BRANCHES]
        for remote in repo[config.JSON_REMOTES]:
            for _ in range(config.TEST_NUM_ROUNDS):
                LOGGER.info(config.LOG_ROUND_INFO, current_round,
                            total_rounds, os.path.basename(repo_dir), remote)
                #delete_remote_branch(repo_dir, remote, branch)

                existing_rounds = [
                    entry[config.STATS_ROUND]
                    for entry in performance_data.get(os.path.basename(repo_dir),
                                                      {}).get(remote, [])
                ]
                if _ in existing_rounds:
                    LOGGER.info(config.LOG_ROUND_SKIP, _, os.path.basename(repo_dir), remote)
                    current_round += 1
                    continue

                create_and_commit_file(repo_dir, remote)
                git_push(repo_dir, remote, branch, _)

                if remote.lower() == config.GIT_CRYPT_REMOTE:
                    run_command([config.GIT_CRYPT, config.GIT_UNLOCK])

                current_round += 1
                time.sleep(2)

    calculate_statistics(performance_data)
    generate_box_plots(performance_data)
    delete_tmp_directory_contents()

if __name__ == "__main__":
    main()
