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
import json
import subprocess
import statistics
import threading
from datetime import datetime
from functools import wraps
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import config
import logger

performance_data = {}
LOGGER = logger.configure_logger()

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

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=metric_df, x=config.STATS_REPODIR,
                    y=config.STATS_VALUE, hue=config.STATS_REMOTE)

        plt.title(f"{metric.replace('_', ' ').title()}")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.ylim(bottom=0)
        plt.grid(True)
        plt.tight_layout()

        timestamp = datetime.now().strftime(config.DATETIME_FORMAT)
        filename = f"outputs/{timestamp}_{metric}_comparison.png"
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
                                     in performance_data[repo_dir][remote]
                                     if entry.get(config.STATS_ROUND) == round_num), None)

        if existing_round_entry is not None:
            existing_round_entry.update({
                config.STATS_EXECUTION_TIME: execution_time,
                config.STATS_NET_SENT: net_sent,
                config.STATS_NET_RECEIVED: net_recv,
                config.STATS_DISK_READ: disk_read,
                config.STATS_DISK_WRITE: disk_write
            })

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


def delete_remote_repo(directory, remote, branch):
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
        time.sleep(0.1)
    return max_memory_usage[0]


def monitor_cpu(pids, tot_cpu_usage, rounds, process):
    """Monitors the CPU usage in a separate thread."""
    while process.poll() is None:
        cpu, r = get_cpu_usage(pids)
        tot_cpu_usage[0] += cpu
        rounds[0] += r
        time.sleep(0.1)
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
def git_push(directory, remote, branch, i):
    """Performs a git push in the specified directory."""
    if not os.path.isdir(directory):
        LOGGER.error(config.LOG_DIRECTORY, directory)
        return

    try:
        os.chdir(directory)

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

        if directory not in performance_data:
            performance_data[directory] = {}

        if remote not in performance_data[directory]:
            performance_data[directory][remote] = []

        performance_data[directory][remote].append({
            config.STATS_ROUND: i,
            config.STATS_PEAK_MEMORY: max_memory_usage,
            config.STATS_AVG_CPU: avg_cpu_usage
        })

    except subprocess.CalledProcessError as e:
        LOGGER.error(config.LOG_PUSH_FAIL, e.stderr)
    except Exception as e:
        LOGGER.error(config.LOG_ERROR, e)


def calculate_statistics(data):
    """Calculates statistics (average & standard deviation) from
    the collected performance data and writes to a file."""
    if not data:
        LOGGER.warning(config.ERROR_NO_STATS)
        return

    timestamp = datetime.now().strftime(config.DATETIME_FORMAT)
    filename = f"outputs/{timestamp}_stats.txt"

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open(filename, config.FILE_WRITE, encoding=config.FILE_UTF8) as file:
        for repo_dir, remotes in data.items():
            LOGGER.info(config.LOG_STATS_REPO, os.path.basename(repo_dir))
            file.write(f"\nStatistics for Repository: {os.path.basename(repo_dir)}\n")

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

                LOGGER.info(output)
                file.write(output)

        file.write(config.LOG_RAWDATA)
        json.dump(data, file, indent=4)

    LOGGER.debug(config.LOG_STATS_WRITE, filename)


def main():
    """Main method to conduct all measurements over all repos and remotes"""
    repo_data = read_repository_information(config.JSON_REPOSITORIES)
    total_rounds = sum(len(repo[config.JSON_REMOTES])
                       for repo in repo_data) * config.TEST_NUM_ROUNDS

    current_round = 1
    for repo in repo_data:
        repo_dir = repo[config.JSON_REPODIR]
        branch = repo[config.JSON_BRANCHES]
        for remote in repo[config.JSON_REMOTES]:
            for i in range(config.TEST_NUM_ROUNDS):
                LOGGER.info(config.LOG_ROUND_INFO, current_round,
                            total_rounds, os.path.basename(repo_dir), remote)
                delete_remote_repo(repo_dir, remote, branch)
                git_push(repo_dir, remote, branch, i)
                if remote.lower() == config.GIT_CRYPT_REMOTE:
                    run_command([config.GIT_CRYPT, config.GIT_UNLOCK])
                current_round += 1

    calculate_statistics(performance_data)
    generate_box_plots(performance_data)

if __name__ == "__main__":
    main()
