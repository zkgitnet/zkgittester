import logging

# JSON Configuration
JSON_REPOSITORIES = "repositories.json"
JSON_REPODIR = "repoDir";
JSON_BRANCHES = "branch";
JSON_REMOTES = "remotes";
JSON_REPOS = "repositories";

# Git Configuration
GIT_GIT = "git"
GIT_PUSH = "push"
GIT_DELETE = "--delete"
GIT_FORCE = "--force"

# Test Configuration
TEST_NUM_ROUNDS = 2
PID_TERMS = ["git", "gcrypt"]
PID = "pid"
CMDLINE = "cmdline"

# Logger Configuration
LOGGING_LEVEL = logging.DEBUG


# Stats Configuration
STATS_DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"
STATS_MB = 1024 * 1024
STATS_EXECUTION_TIME = "execution_time"
STATS_NET_SENT = "net_sent"
STATS_NET_RECEIVED = "net_recv"
STATS_DISK_READ = "disk_read"
STATS_DISK_WRITE = "disk_write"
STATS_ROUND = "round"
STATS_PEAK_MEMORY = "peak_memory"

STATS = [STATS_EXECUTION_TIME, STATS_PEAK_MEMORY, STATS_NET_SENT,
         STATS_NET_RECEIVED, STATS_DISK_READ, STATS_DISK_WRITE]

# Error Messages
ERROR_NO_STATS = "No data to calculate statistics"
ERROR_NO_PID = "No PIDs found or access denied"
