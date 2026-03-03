# IDE Agent Operational Guidelines

## 1. Core Directives
You are the primary developer agent for my computational physics course project. Your primary objective is to write robust, mathematically sound, and maintainable Python code for our econophysics research.

**You must strictly adhere to the Test-Driven Development (TDD) paradigm.** Do not write implementation code before writing the tests that define its expected behavior.

## 2. The TDD Workflow (Red-Green-Refactor)
For every task, issue, or feature request, you must execute the following loop:

1. **Write the Test (Red):** Create a test in the `tests/` directory for the specific function or method you are about to build.
2. **Run the Test:** Execute `pytest` to confirm the test fails. (This validates that the test is actually testing something new).
3. **Write the Code (Green):** Write the absolute minimum implementation code required in the `src/` directory to make the test pass.
4. **Run the Test Again:** Execute `pytest` to confirm the test now passes.
5. **Refactor:** Clean up the code, optimize imports, and improve docstrings without changing behavior.

## 3. Mandatory Quality Gates
Before you declare a task "complete" or stop your generation loop, you must run and pass the following three checks. **Do not ask for permission to run these; execute them automatically.**

* **Testing:** Run `pytest tests/`. All tests must pass.
* **Type Checking:** Run `mypy src/`. There must be zero type errors. Ensure all new functions have strict Python 3.11+ type hints.
* **Linting & Formatting:** Run `ruff check src/ tests/ --fix` and `ruff format src/ tests/`. 
