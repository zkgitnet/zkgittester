"""
Configuration Constants for zkGit Testing Framework

This module defines constants used across the zkGit test suite. It centralizes configuration
for JSON keys, Git commands, test parameters, logging, performance statistics, and error messages.
"""

import logging

# JSON Configuration
JSON_REPOSITORIES = "repositories.json"
JSON_REPODIR = "repo_dir"
JSON_BRANCHES = "branch"
JSON_REMOTES = "remotes"
JSON_REPOS = "repositories"

# Git Configuration
GIT_GIT = "git"
GIT_PUSH = "push"
GIT_PULL = "pull"
GIT_CLONE = "clone"
GIT_ADD = "add"
GIT_COMMIT = "commit"
GIT_MESSAGE = "-m"
GIT_DELETE = "--delete"
GIT_FORCE = "--force"
GIT_CRYPT = "git-crypt"
GIT_LOCK = "lock"
GIT_UNLOCK = "unlock"
OS_RM = "rm"
OS_RF = "-rf"

# Test Configuration
TEST_NUM_ROUNDS = 30
SHOW_PLOTS = False
TEST_COMMANDS = ["clone", "push", "pull"]
PID_TERMS = ["git", "gcrypt"]
PID = "pid"
CMDLINE = "cmdline"
GIT_CRYPT_REMOTE = "gitcryptremote"
GIT_REMOTE = "gitremote"
GIT_GCRYPT_REMOTE = "gcryptremote"
GIT_ZKGIT_REMOTE = "zkgitremote"
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"

# File Configuration
FILE_READ = "r"
FILE_WRITE = "w"
FILE_WRITE_BINARY = "wb"
FILE_UTF8 = "utf-8"
FILE_JSON = ".json"
DIR_TMP = "tmp"

# Logger Configuration
LOGGING_LEVEL = logging.DEBUG
LABEL_DECRYPT = ""
LABEL_ENCRYPT = "Encryption"
LABEL_REMOVE_DIR = "Deleting directory"
LABEL_PROCESS = "Process"


LOG_BOXPLOT = "Boxplot saved as %s"
LOG_DIRECTORY = "Error: Directory '%s' does not exist"
LOG_RESET_REPO = "Resetting remote repo. PID: %d"
LOG_RUN_COMMAND = "Running command: %s (PID: %d)"
LOG_FILE_NOT_FOUND = "Filename %s does not match expected pattern, skipping"
LOG_PROCESS_FOUND = "[%s] Relevant processes found: %s"
LOG_STATS_WRITE = "\nStatistics written to: %s"
LOG_ROUND_INFO = "Round %d of %d: %s at %s"
LOG_ROUND_SKIP = "Skipping existing round %d for %s/%s"
LOG_FILE_DELETE_FAIL = "Error deleting file %s: %s"
LOG_DELETE_FAIL = "Delete failed: %s"
LOG_ANOVA = "ANOVA result for %s: F-statistic = %.2f, p-value = %.8f"
LOG_TTEST = "T-test result for %s vs %s (%s): t-statistic = %.2f, p-value = %.8f"
LOG_TTEST_GLOBAL = "%s: %s vs %s: t = %.2f, p = %.8f"
LOG_TTEST_FAIL = "T-test skipped for {%s} vs {%s} (%s) due to missing data"
LOG_METRIC_MISSING = "Metric '%s' is missing for one or more remotes in repository %s"
LOG_PUSH_FAIL = "Git push failed: %s"
LOG_COMMIT = "Created and committed 1KB file: %s"
LOG_COMMIT_FAIL = "Git commit failed: %s"
LOG_STATS_REPO = "\nStatistics for Repository: %s"
LOG_ERROR = "Unexpected error: %s"
LOG_RAWDATA = "\nRaw Data:\n"
LOG_ROUND_DATA = "[%s | %s | %s]"
LOG_ROUND_TIME = "  Execution time: %.2f seconds"
LOG_ROUND_NET = "  Network sent: %.2f MB, received: %.2f MB"
LOG_ROUND_DISK = "  Disk read: %.2f MB, written: %.2f MB"
LOG_ROUND_MEM = "  Memory usage: %.2f MB"
LOG_ROUND_CPU = "  CPU usage: %.2f %%"

# Stats Configuration
STATS_DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"
STATS_MB = 1024 * 1024
STATS_EXECUTION_TIME = "execution_time"
STATS_AVG_CPU = "avg_cpu"
STATS_NET_SENT = "net_sent"
STATS_NET_RECEIVED = "net_recv"
STATS_DISK_READ = "disk_read"
STATS_DISK_WRITE = "disk_write"
STATS_ROUND = "round"
STATS_PEAK_MEMORY = "peak_memory"
STATS_REPODIR = "repo_dir"
STATS_REMOTE = "remote"
STATS_METRIC = "metric"
STATS_VALUE = "value"

STATS = [STATS_EXECUTION_TIME, STATS_PEAK_MEMORY, STATS_NET_SENT,
         STATS_NET_RECEIVED, STATS_DISK_READ, STATS_DISK_WRITE,
         STATS_AVG_CPU]

# Error Messages
ERROR_NO_STATS = "No data to calculate statistics"
ERROR_NO_PID = "No PIDs found or access denied"
ERROR_NO_BOXPLOT = "No data to generate boxplots"
ERROR_NO_METRICS = "No valid matrics found to plot"
