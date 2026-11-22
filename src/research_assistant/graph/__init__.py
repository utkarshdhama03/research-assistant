"""Research graph (LangGraph). Importing this package switches LangSmith tracing off.

langgraph depends on langchain-core, whose langsmith client would send traces to an external
service if LANGSMITH_TRACING / LANGCHAIN_TRACING_V2 were set. Nothing leaves this machine, so both
are forced off before langgraph is imported anywhere in this package.
"""

import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
