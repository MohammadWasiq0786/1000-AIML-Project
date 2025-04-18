"""
Project 901. Dynamic Code Analysis

Dynamic code analysis monitors a program while it's running to detect issues like memory leaks, unhandled exceptions, or insecure behavior. In this project, we simulate runtime monitoring of Python code by capturing function calls, exceptions, and execution time using decorators.

What This Does:
Tracks execution time of functions

Logs exceptions with tracebacks

Can be extended to monitor memory usage, API calls, or file access

🧪 Real-world dynamic analysis tools include:

Python tracers (sys.settrace, cProfile, Py-Spy)

Instrumentation frameworks (e.g., Dynatrace, Valgrind, AppDynamics)

Integration with test coverage and fuzzing tools
"""

import time
import traceback
 
# Decorator to analyze function execution
def dynamic_analyzer(func):
    def wrapper(*args, **kwargs):
        print(f"\n[Analyzing] Function: {func.__name__}")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            print(f"[OK] Execution Time: {duration:.4f} seconds")
            return result
        except Exception as e:
            duration = time.time() - start_time
            print(f"[ERROR] Exception in {func.__name__} after {duration:.4f} seconds")
            print("→", traceback.format_exc().strip())
    return wrapper
 
# Simulated target functions
@dynamic_analyzer
def safe_function(x):
    time.sleep(0.5)
    return x * 2
 
@dynamic_analyzer
def risky_function(y):
    time.sleep(0.3)
    return y / 0  # triggers ZeroDivisionError
 
# Run analysis
safe_function(10)
risky_function(5)