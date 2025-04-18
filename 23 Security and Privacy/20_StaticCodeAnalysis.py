"""
Project 900. Static Code Analysis

Static code analysis examines source code without executing it to detect potential bugs, code smells, security vulnerabilities, and style violations. In this project, we simulate a static code analysis tool that scans Python files for risky patterns (like use of eval, hardcoded passwords, or missing exception handling).

What It Detects:
Dangerous use of eval()

Hardcoded credentials

Unsafe operations (like divide by zero)

Lack of exception handling scaffolding (basic)

🛡️ In real tools:

Use AST parsers (ast, pylint, bandit, flake8) for structured analysis

Export reports as HTML/JSON for CI pipelines

Extend to language-specific linters and security scanners
"""

import re
 
# Sample Python code (can also load from file)
code_sample = """
def calculate(expression):
    result = eval(expression)
    return result
 
password = '123456'  # hardcoded password
 
def risky_function():
    x = 10 / 0  # division by zero risk
"""
 
# Define patterns to detect
patterns = {
    'Use of eval()': r'\beval\(',
    'Hardcoded password': r'password\s*=\s*[\'"].+[\'"]',
    'Division by zero': r'/\s*0',
    'Missing exception handling': r'def\s+\w+\([^)]*\):\s*\n\s*[^\s]'
}
 
# Scan the code line by line
print("Static Code Analysis Report:")
lines = code_sample.split('\n')
for i, line in enumerate(lines, 1):
    for issue, pattern in patterns.items():
        if re.search(pattern, line):
            print(f"[Line {i}] ⚠️ {issue} → {line.strip()}")