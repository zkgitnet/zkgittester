import config
import logger

import os
import json
import subprocess
import psutil
import time
import logging
import statistics

from functools import wraps
from datetime import datetime

performanceData = {}
LOGGER = logger.configure_logger()

def findProcesses(terms):
    """Finds all process PIDs where the command contains any of the given search strings from the list of terms."""
    matchingPids = []
    for process in psutil.process_iter(attrs=[config.PID, config.CMDLINE]):
        try:
            cmdline = process.info[config.CMDLINE]
            if cmdline and any(term in cmd for term in terms for cmd in cmdline):
                matchingPids.append(process.info[config.PID])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            LOGGER.warning(config.ERROR_NO_PID)
            continue

    return matchingPids


def getMemoryUsage(pidList):
    """Returns the total memory usage (in MB) for all given PIDs, including their child processes."""
    
    totalMemory = 0.0
    for pid in pidList:
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            allProcesses = [parent] + children 
            
            totalMemory += sum(p.memory_info().rss for p in allProcesses if p.is_running()) / config.STATS_MB
        except psutil.NoSuchProcess:
            continue 

    return totalMemory

def measurePerformance(func):
    """Decorator to measure execution time, memory usage, and I/O stats of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()

        startTime = time.time()
        netBefore = psutil.net_io_counters() 
        diskBefore = psutil.disk_io_counters() 

        result = func(*args, **kwargs)

        executionTime = time.time() - startTime
        netAfter = psutil.net_io_counters()  
        diskAfter = psutil.disk_io_counters() 

        netSent = (netAfter.bytes_sent - netBefore.bytes_sent) / config.STATS_MB
        netRecv = (netAfter.bytes_recv - netBefore.bytes_recv) / config.STATS_MB
        diskRead = (diskAfter.read_bytes - diskBefore.read_bytes) / config.STATS_MB
        diskWrite = (diskAfter.write_bytes - diskBefore.write_bytes) / config.STATS_MB

        repoDir = args[0] 
        remote = args[1]  
        roundNum = args[3]

        existingRoundEntry = next((entry for entry in performanceData[repoDir][remote] if entry.get(config.STATS_ROUND) == roundNum), None)

        if existingRoundEntry is not None:
            existingRoundEntry.update({
                config.STATS_EXECUTION_TIME: executionTime,
                config.STATS_NET_SENT: netSent,
                config.STATS_NET_RECEIVED: netRecv,
                config.STATS_DISK_READ: diskRead,
                config.STATS_DISK_WRITE: diskWrite
            })

        LOGGER.debug(f"[{os.path.basename(repoDir)} | {remote} | Round {roundNum}]")
        LOGGER.debug(f"  Execution time: {executionTime:.2f} seconds")
        LOGGER.debug(f"  Network sent: {netSent:.2f} MB, received: {netRecv:.2f} MB")
        LOGGER.debug(f"  Disk read: {diskRead:.2f} MB, written: {diskWrite:.2f} MB")
        LOGGER.debug(f"  Memory usage: {existingRoundEntry.get('peak_memory'):.2f} MB")

        return result

    return wrapper

def readRepositoryInformation(jsonFile):
    with open(jsonFile, "r") as inputFile:
        data = json.load(inputFile)
    return data[config.JSON_REPOS]


def deleteRemoteRepo(directory, remote, branch):
    """Deletes all the remote content from the specified directory."""
    if not os.path.isdir(directory):
        LOGGER.error(f"Error: Directory '{directory}' does not exist.")
        return
    
    try:
        os.chdir(directory)
        process = subprocess.Popen(
            [config.GIT_GIT, config.GIT_PUSH, config.GIT_DELETE, remote, branch],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        LOGGER.debug(f"Resetting remote repo. PID: {process.pid}")
        process.wait()
    except subprocess.CalledProcessError as e:
        LOGGER.warning(f"Delete failed: {e.stderr}")
    except Exception as e:
        LOGGER.critical(f"Unexpected error: {e}")    

        
@measurePerformance
def gitPush(directory, remote, branch, i):
    """Performs a git push in the specified directory."""
    if not os.path.isdir(directory):
        LOGGER.error(f"Error: Directory '{directory}' does not exist.")
        return
    
    try:
        os.chdir(directory)
        process = subprocess.Popen(
            [config.GIT_GIT, config.GIT_PUSH, remote, branch, config.GIT_FORCE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        LOGGER.info(f"Git push for {os.path.basename(directory)} started with PID: {process.pid}")

        maxMemoryUsage = 0.0
        output = True
        while process.poll() is None: 
            pids = findProcesses(config.PID_TERMS)
            if output:
                LOGGER.debug(f"Relevant processes found {pids}")
                output = False

            maxMemoryUsage = max(maxMemoryUsage, getMemoryUsage(pids))
            time.sleep(0.2)

        if directory not in performanceData:
            performanceData[directory] = {}

        if remote not in performanceData[directory]:
            performanceData[directory][remote] = []
            
        performanceData[directory][remote].append({
            config.STATS_ROUND: i,
            config.STATS_PEAK_MEMORY: maxMemoryUsage
        })

        process.wait()
        
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"Git push failed: {e.stderr}")
    except Exception as e:
        LOGGER.error(f"Unexpected error: {e}")

        
def calculateStatistics(data):
    """Calculates statistics (average & standard deviation) from the collected performance data and writes to a file."""
    if not data:
        LOGGER.warning("No data to calculate statistics.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_stats.txt"

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open(filename, "w") as file:
        for repoDir, remotes in data.items():
            LOGGER.info(f"\nStatistics for Repository: {os.path.basename(repoDir)}")
            file.write(f"\nStatistics for Repository: {os.path.basename(repoDir)}\n")

            for remote, entries in remotes.items():
                numRuns = len(entries)

                if numRuns < 2:
                    LOGGER.warning(f"\n  Remote: {remote} (Total Runs: {num_runs}) - Not enough data for standard deviation")
                    file.write(f"\n  Remote: {remote} (Total Runs: {num_runs}) - Not enough data for standard deviation\n")
                    continue
                
                metricValues = {metric: [entry[metric] for entry in entries if metric in entry] for metric in config.STATS}

                avgValues = {metric: sum(values) / numRuns for metric, values in metricValues.items()}
                stdValues = {
                    metric: statistics.stdev(values) if len(values) > 1 else 0
                    for metric, values in metricValues.items()
                }

                output = f"\n  Remote: {remote} (Total Runs: {numRuns})\n"
                for metric in config.STATS:
                    unit = "sec" if metric == config.STATS_EXECUTION_TIME else "MB"
                    output += (
                        f"  Avg {metric.replace('_', ' ').title()}: {avgValues[metric]:.2f} "
                        f"{unit}, Std Dev: {stdValues[metric]:.2f} {unit}\n"
                    )

                LOGGER.info(output)
                file.write(output)

    LOGGER.debug(f"\nStatistics written to: {filename}")


def main():
    repoData = readRepositoryInformation(config.JSON_REPOSITORIES);
    totalRounds = sum(len(repo[config.JSON_REMOTES]) for repo in repoData) * config.TEST_NUM_ROUNDS

    currentRound = 1
    for repo in repoData:
        repoDir = repo[config.JSON_REPODIR]
        branch = repo[config.JSON_BRANCHES]
        for remote in repo[config.JSON_REMOTES]:
            for i in range(config.TEST_NUM_ROUNDS):
                LOGGER.info(f"Round {currentRound} of {totalRounds}: {os.path.basename(repoDir)} at {remote}")
                deleteRemoteRepo(repoDir, remote, branch)
                gitPush(repoDir, remote, branch, i)
                currentRound += 1

    calculateStatistics(performanceData)
    
if __name__ == "__main__":
    main()

