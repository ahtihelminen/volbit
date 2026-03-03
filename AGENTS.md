# IDE Agent Operational Guidelines

## 1. Core Directives
You are the primary developer agent for my computational physics course project. Your primary objective is to write robust, mathematically sound, and maintainable Python code for our econophysics research.

**You must strictly adhere to the Test-Driven Development (TDD) paradigm.** Do not write implementation code before writing the tests that define its expected behavior.



## 2. The TDD Workflow (Red-Green-Refactor)
For every task, issue, or feature request, you must execute the following loop:

1. **Write the Test (Red):** Create a test in the `tests/` directory for the specific function or method you are about to build.
2. **Run the Test:** Execute `uv run pytest` to confirm the test fails. (This validates that the test is actually testing something new).
3. **Write the Code (Green):** Write the absolute minimum implementation code required in the `src/` directory to make the test pass.
4. **Run the Test Again:** Execute `uv run pytest` to confirm the test now passes.
5. **Refactor:** Clean up the code, optimize imports, and improve docstrings without changing behavior.

## 3. Mandatory Quality Gates
Before you declare a task "complete" or stop your generation loop, you must run and pass the following three checks. **Do not ask for permission to run these; execute them automatically.**

* **Testing:** Run `uv run pytest tests/`. All tests must pass.
* **Type Checking:** Run `mypy src/`. There must be zero type errors. Ensure all new functions have strict Python 3.11+ type hints. Do not add ignore flags to type erros.
* **Linting & Formatting:** Run `ruff check src/ tests/ --fix` and `ruff format src/ tests/`. 

## 4. GitHub Workflow (CRITICAL: DO NOT SKIP)
**You are NOT done until the code is merged and the issue is closed.**

For every feature or fix, you **MUST** follow this exact sequence:

1.  **Branch:** Create a new branch from `main` with a descriptive name (e.g., `feat/add-model`, `fix/typo`).
2.  **Implement:** Follow the TDD workflow (Section 2) and make small, meaningful commits.
3.  **Verify:** Run the Mandatory Quality Gates (Section 3).
4.  **Pull Request:**
    *   Push the branch: `git push origin <branch-name>`
    *   Create a PR: `gh pr create --title "..." --body "..." --head <branch-name> --base main`
    *   Merge the PR: `gh pr merge <pr-number> --merge --delete-branch`
5.  **Issue Status:**
    *   **Start:** Move issue to 'In Progress' (if applicable).
    *   **Finish:** Close the issue using `gh issue close <issue-number>`.