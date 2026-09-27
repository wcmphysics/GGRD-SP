# GGRD-SP Project Context
This is a sequence-to-sequence Machine Learning project that try to find how to transform a spectrum meassured at a tool to another spectrum as if the specimen were measured at another tool. Namely, we want to use machine learning model to capture tool difference in spectra so we can transfer spectra between tools. This project relies heavily on PyTorch for neural networks and Pandas/SciPy for data processing.

## Tech Stack
- Python 3
- Scikit-learn, PyTorch, Pandas, NumPy, SciPy
- Jupyter Notebooks for exploration, Python scripts for utilities.

## Agent Instructions (Rules for Gemini)
1. **Virtual Environment:** Always execute Python commands using the local environment at `\.venv\Scripts\python.exe`.
2. **File Structure:** 
   - Main python body: `main.py`
   - Core ML models and utilities belong in the `models/` and `utility/` folders.
3. **Code Style:** Use strict PEP-8 formatting. Include type hints and concise docstrings for all new functions.
4. **Git Commits and Pushing:** When making commits and push, you MUST ALWAYS show me the commit message before any commit and push. Only after I approve will you do the commits and push.
5. **Commit Size:** When the change is more than 50 lines, try to separate the commits and so one commit is readable and will not be flooded with changes.
6. **Modular Functionality:** 
   - Implement new features in a strictly modular fashion inside dedicated modules within `models/` or `utility/`.
   - New functionality must be encapsulated as callable functions or classes ready for import by `main.py`, avoiding unsolicited edits or side effects to `main.py` or other existing modules.
7. **Clean Function Interfaces (Parameter Packaging):**
   - If a function requires 5 or more parameters (>= 5), bundle them into a readable configuration dictionary (e.g., `config: dict[str, Any]`).
   - Dictionaries are strongly preferred by default for readability and simplicity. A dedicated dataclass should only be used when there is a clear, tangible advantage over a dictionary (such as complex nested hierarchies or strict validation requirements).
8. **Existing Code Modifications & Prior Approval:**
   - If modifying existing code is unavoidable, always inform the user of what will be changed and why *before* touching the files.
   - Verify and guarantee that the modifications do not break existing functionality or workflows.
9. **Multi-Agent Review and Test Loop:**
   - When writing new code or significant components, launch a separate reviewer subagent to critique the code, probe for edge cases, inspect error handling, and run verification tests.
   - Address the reviewer agent's feedback and refine the implementation.
   - Repeat this write-test-rewrite loop up to 3 iterations until the code is robust and passes all checks.