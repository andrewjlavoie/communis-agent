from datetime import timedelta

from temporalio.common import RetryPolicy

LLM_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=5,
)

# Timeouts — LLM calls can take minutes for long context; file I/O is fast
LLM_TIMEOUT = timedelta(minutes=30)
FAST_LLM_TIMEOUT = timedelta(minutes=10)
IO_TIMEOUT = timedelta(seconds=30)
TOOL_TIMEOUT = timedelta(minutes=5)
