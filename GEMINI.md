# GGRD-SP Project Context
This is a sequence-to-sequence Machine Learning project that try to find how to transform a spectrum meassured at a tool to another spectrum as if the specimen were measured at another tool. Namely, we want to use machine learning model to capture tool difference in spectra so we can transfer spectra between tools. This project relies heavily on PyTorch for neural networks and Pandas/SciPy for data processing.

## Program Structure
The codes will be composed of mainly 5 parts
1. Spectrum data reading or generation
   - there are many kinds of materials for the specimen under XPS measurement
   - 1 XSP measurement gives many (specified by `N_die` and defaulted to `9`) full spectra
   - 1 full spectrum contains many (specified by `N_region` depending on material) regions (or called regional spectra)
   - one region has `N_points` data points
   - store the regional spectrum data (arrays of intensity and binding energy) and meta data separately
   - store regional spectrum intensity as 2d numpy array `ary_intensity` where the nth row is the nth intensity array
   - store regional bidning energy as 2d numpy array `ary_energy` where the nth row is the nth binding energy array
   - Note that one array in the `ary_intensity` or `ary_energy` is one regional spectrum for one specific material, one specific measurement, one specific die, and one specific region. The shape of `ary_intensity` or `ary_energy` will be (`N_measurement`*`N_die`*`N_region`, `N_points`).
   - meta data is a pandas dataframe containing all meta data of the spectra recorded
   - meta data includes: material, tool and time of measurement, die, region, number of points in one spectrum, and the corresponding index in the intensity and energy array
2. Pairing source and target spectra
   - allow users to specify source tool and target tool
   - for every measurement in source tool, we find/map a measurement from the target tool
   - this source to target mapping must be 1-to-1 (no multiple source measurement is linked to the same target measurement)
   - the time difference between source and target measurement should be lower than a user-specified threshold (default to 12 hours) 
   - store this pairing information in the meta data dataframe (add columns like tool_target, measurement_id_target, spectrum_index_target)
3. Neural network definition, training, and hyperparamter search (by Bayesian optimization)
4. Model performance observation
5. Calculation of atomic percentage for elements by area integration

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
4. **Autonomous Commits and Pushing:**
   - Once a feature request, plan, or review iteration is approved, autonomously commit and push the changes without asking for separate micro-permissions on commit messages.
   - Follow standard Conventional Commits format (e.g. `feat:`, `fix:`, `test:`, `docs:`) and always display the created commit hashes and messages in your final summary for transparency.
5. **Commit Size:** When the change is more than 50 lines, try to separate the commits and so one commit is readable and will not be flooded with changes.
6. **Modular Functionality:** 
   - Implement new features in a strictly modular fashion inside dedicated modules within `models/` or `utility/`.
   - New functionality must be encapsulated as callable functions or classes ready for import by `main.py`, avoiding unsolicited edits or side effects to `main.py` or other existing modules.
7. **Clean Function Interfaces (Parameter Packaging):**
   - If a function requires 5 or more parameters (>= 5), bundle them into a readable configuration dictionary (e.g., `config: dict[str, Any]`).
   - Dictionaries are strongly preferred by default for readability and simplicity. A dedicated dataclass should only be used when there is a clear, tangible advantage over a dictionary (such as complex nested hierarchies or strict validation requirements).
8. **Existing Code Modifications & Safeguards:**
   - When modifying existing code is required, briefly state what will change in your explanation. Once given the go-ahead, proceed through implementation, testing, and commits without redundant confirmation rounds.
   - Verify and guarantee that the modifications do not break existing functionality or workflows.
9. **Multi-Agent Review and Test Loop:**
   - When writing new code or significant components, launch a separate reviewer subagent to critique the code, probe for edge cases, inspect error handling, and run verification tests.
   - Address the reviewer agent's feedback and refine the implementation.
   - Repeat this write-test-rewrite loop up to 3 iterations until the code is robust and passes all checks.