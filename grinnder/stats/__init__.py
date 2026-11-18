"""Statistics module for emasureing/tracing latency"""

from grinnder.stats.stats import stat, timed, timed_by
from grinnder.stats.trace import trace

__all__ = ["stat", "timed", "timed_by", "trace"]
