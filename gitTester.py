import config

import os
import json
import subprocess
import psutil
import time
import statistics

from functools import wraps
from datetime import datetime

performanceData = {}

def find_processes_by_names(term1, term2):
    """Finds all process PIDs where the command contains either of the given search strings."""
    matching_pids = []
    for process in psutil.process_iter(attrs=['pid', 'cmdline']):
        try:
            cmdline = process.info['cmdline']
            if cmdline and any(term in cmd for term in (term1, term2) for cmd in cmdline):
                matching_pids.append(process.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue 

    return matching_pids

def get_memory_usage_for_pids(pid_list):
    """Returns the total memory usage (in MB) for all given PIDs, including their child processes."""
    total_memory = 0.0

    for pid in pid_list:
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            all_processes = [parent] + children 
            
            total_memory += sum(p.memory_info().rss for p in all_processes if p.is_running()) / (1024 * 1024)
        except psutil.NoSuchProcess:
            continue 

    return total_memory

def measurePerformance(func):
    """Decorator to measure execution time, memory usage, and I/O stats of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()

        start_time = time.time()
        net_before = psutil.net_io_counters() 
        disk_before = psutil.disk_io_counters() 

        result = func(*args, **kwargs)

        execution_time = time.time() - start_time
        net_after = psutil.net_io_counters()  
        disk_after = psutil.disk_io_counters() 

        net_sent = (net_after.bytes_sent - net_before.bytes_sent) / (1024 * 1024)  # Sent MB
        net_recv = (net_after.bytes_recv - net_before.bytes_recv) / (1024 * 1024)  # Received MB
        disk_read = (disk_after.read_bytes - disk_before.read_bytes) / (1024 * 1024)  # Read MB
        disk_write = (disk_after.write_bytes - disk_before.write_bytes) / (1024 * 1024)  # Written MB

        repo_dir = args[0] 
        remote = args[1]  
        round_num = args[3] 

        if repo_dir not in performanceData:
            performanceData[repo_dir] = {}

        if remote not in performanceData[repo_dir]:
            performanceData[repo_dir][remote] = []

        performanceData[repo_dir][remote].append({
            "round": round_num,
            "execution_time": execution_time,
            "net_sent": net_sent,
            "net_recv": net_recv,
            "disk_read": disk_read,
            "disk_write": disk_write
        })

        print(f"[{repo_dir} | {remote} | Round {round_num}]")
        print(f"  Execution time: {execution_time:.2f} seconds")
        print(f"  Network sent: {net_sent:.2f} MB, received: {net_recv:.2f} MB")
        print(f"  Disk read: {disk_read:.2f} MB, written: {disk_write:.2f} MB")

        return result

    return wrapper

def readRepositoryInformation(jsonFile):
    with open(jsonFile, "r") as inputFile:
        data = json.load(inputFile)
    return data["repositories"]


def deleteRemoteRepo(directory, remote, branch):
    """Deletes all the remote content from the specified directory."""
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    try:
        os.chdir(directory)
        subprocess.run(["git", "push", "--delete", remote, branch], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Delete failed: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error: {e}")    

        
@measurePerformance
def gitPush(directory, remote, branch, i):
    """Performs a git push in the specified directory."""
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    try:
        os.chdir(directory)
        process = subprocess.Popen(
            ["git", "push", remote, branch],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print(f"Git push started with PID: {process.pid}")

        memory_usages = []
        while process.poll() is None: 
            search_term1 = "git"
            search_term2 = "gcrypt"
            pids = find_processes_by_names(search_term1, search_term2)

            print(f"Processes containing '{search_term1}' or '{search_term2}': {pids}")
            memory_used = get_memory_usage_for_pids(pids)
            memory_usages.append(memory_used)
            print(f"Current Memory Usage: {memory_used:.2f} MB")
            time.sleep(1)

        # Get final memory usage after process exits
        print(f"Final Memory Usage: {max(memory_usages):.2f} MB")

        performanceData[directory][remote].append({
            "peak_memory": max(memory_usages),
        })

        stdout, stderr = process.communicate() 
    except subprocess.CalledProcessError as e:
        print(f"Git push failed: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error: {e}")

        
def calculateStatistics(data):
    """Calculates statistics (average & standard deviation) from the collected performance data and writes to a file."""
    if not data:
        print("No data to calculate statistics.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_stats.txt"

    with open(filename, "w") as file:
        for repo_dir, remotes in data.items():
            print(f"\nStatistics for Repository: {repo_dir}")
            file.write(f"\nStatistics for Repository: {repo_dir}\n")

            for remote, entries in remotes.items():
                num_runs = len(entries)

                if num_runs < 2:
                    print(f"\n  Remote: {remote} (Total Runs: {num_runs}) - Not enough data for standard deviation")
                    file.write(f"\n  Remote: {remote} (Total Runs: {num_runs}) - Not enough data for standard deviation\n")
                    continue

                execution_times = [entry["execution_time"] for entry in entries]
                peak_memories = [entry["peak_memory"] for entry in entries]
                net_sent_values = [entry["net_sent"] for entry in entries]
                net_recv_values = [entry["net_recv"] for entry in entries]
                disk_read_values = [entry["disk_read"] for entry in entries]
                disk_write_values = [entry["disk_write"] for entry in entries]

                avg_time = sum(execution_times) / num_runs
                avg_memory = sum(peak_memories) / num_runs
                avg_net_sent = sum(net_sent_values) / num_runs
                avg_net_recv = sum(net_recv_values) / num_runs
                avg_disk_read = sum(disk_read_values) / num_runs
                avg_disk_write = sum(disk_write_values) / num_runs

                std_time = statistics.stdev(execution_times) if num_runs > 1 else 0
                std_memory = statistics.stdev(peak_memories) if num_runs > 1 else 0
                std_net_sent = statistics.stdev(net_sent_values) if num_runs > 1 else 0
                std_net_recv = statistics.stdev(net_recv_values) if num_runs > 1 else 0
                std_disk_read = statistics.stdev(disk_read_values) if num_runs > 1 else 0
                std_disk_write = statistics.stdev(disk_write_values) if num_runs > 1 else 0

                output = (
                    f"\n  Remote: {remote} (Total Runs: {num_runs})\n"
                    f"  Avg Execution Time: {avg_time:.2f} sec, Std Dev: {std_time:.2f} sec\n"
                    f"  Avg Peak Memory Usage: {avg_memory:.2f} MB, Std Dev: {std_memory:.2f} MB\n"
                    f"  Avg Network Sent: {avg_net_sent:.2f} MB, Std Dev: {std_net_sent:.2f} MB\n"
                    f"  Avg Network Received: {avg_net_recv:.2f} MB, Std Dev: {std_net_recv:.2f} MB\n"
                    f"  Avg Disk Read: {avg_disk_read:.2f} MB, Std Dev: {std_disk_read:.2f} MB\n"
                    f"  Avg Disk Write: {avg_disk_write:.2f} MB, Std Dev: {std_disk_write:.2f} MB\n"
                )

                print(output)
                file.write(output)

    print(f"\nStatistics written to: {filename}")

    
if __name__ == "__main__":
    
    repoData = readRepositoryInformation(config.JSON_REPOSITORIES);
    totalRounds = sum(len(repo[config.JSON_REMOTES]) for repo in repoData) * config.TEST_NUM_ROUNDS

    currentRound = 1
    for repo in repoData:
        repoDir = repo[config.JSON_REPODIR]
        branch = repo[config.JSON_BRANCHES]
        for remote in repo[config.JSON_REMOTES]:
            for i in range(config.TEST_NUM_ROUNDS):
                print(f"Round {currentRound} of {totalRounds}: {os.path.basename(repoDir)} at {remote}")
                deleteRemoteRepo(repoDir, remote, branch)
                gitPush(repoDir, remote, branch, i)
                currentRound += 1

    calculateStatistics(performanceData)
