Redis can be used as short-term memory for agent systems.
A common design is to store recent conversation turns, tool traces, and task state in Redis.
Using a sliding window helps control token cost while preserving recent context.