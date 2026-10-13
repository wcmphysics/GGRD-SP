# GGRD-SP Project Context
This is a sequence-to-sequence Machine Learning project that try to find how to transform a spectrum meassured at a tool to another spectrum as if the specimen were measured at another tool. Namely, we want to use machine learning model to capture tool difference in spectra so we can transfer spectra between tools. This project relies heavily on PyTorch for neural networks and Pandas/SciPy for data processing.

## Program Structure
The codes will be composed of the following key stages
1. Spectrum data reading or generation
   - there are many kinds of materials for the specimen under XPS measurement
   - 1 XSP measurement gives many (specified by `N_die` and defaulted to `9`) full spectra
   - 1 full spectrum contains many (specified by `N_region` depending on material) regions (or called regional spectra)
   - one region has `N_points` data points
   - store the regional spectrum data (arrays of intensity and binding energy) and meta data separately
   - store regional spectrum intensity as 2d numpy array `ary_intensity` where the nth row is the nth intensity array
   - store regional bidning energy as 2d numpy array `ary_energy` where the nth row is the nth binding energy array
   - Note that one array in the `ary_intensity` or `ary_energy` is one regional spectrum for one specific material, one specific measurement, one specific die, and one specific region. The shape of `ary_intensity` or `ary_energy` will be (`N_measurement`*`N_die`*`N_region`, `N_points`).
   - meta data (`df_meta`) is a pandas dataframe containing all meta data of the spectra recorded 
   - meta data includes: material, tool and time of measurement, die, region, number of points in one spectrum, and the corresponding index in the intensity and energy array
   - it is possible to have many kinds of materials in the data.
2. Pairing source and target spectra
   - allow users to specify source tool and target tool
   - for every measurement in the source tool, we find/map a measurement
     from the target tool
   - this source to target mapping must be 1-to-1 (no multiple source
     measurement is linked to the same target measurement)
   - the mapping maps measurements from the source tool to the target tool,
     the die and region in that measurement will also be paired accordingly
     (spectrum from die A region B of the source measurement will map to the
     same die A and the same region B in the target measurement)
   - the time difference between source and target measurement should be lower
     than a user-specified threshold (default to 12 hours)
   - store this pairing information in the meta data dataframe (add columns
     like tool_target, measurement_id_target, spectrum_index_target)
3. Normalization of spectral data
   - need a dedicated config for this part
   - inputs are `ary_intensity`, `ary_energy`, and `df_meta`
   - 4 options for performing normalization: by total flux (default),
     regional maximum intensity ('regional_max'), source regional maximum
     intensity ('source_regional_max'), or no normalization ('no_normalization')
   - for normalization by total flux, intensity of every regional spectra from a
     die in a measurement was divided by the total flux of that die in that
     measurement (summation of flux among all regions for that specific die and
     that specific measurement)
   - for normalization by regional maximum intensity, intensity of a regional
     spectrum is divided by the maximum intensity of that specific regional
     spectrum
   - for normalization by source regional maximum intensity
     ('source_regional_max'), this normalization is applied to each region
     individually. For each paired record, both the source regional spectrum
     and its paired target regional spectrum are divided by the maximum
     intensity of that source regional spectrum. Unpaired spectra are not
     scaled (left unscaled)
   - for no normalization ('no_normalization'), intensity array is left unscaled
   - output are `ary_intensity` (normalization applied), `ary_energy`, and
     `df_meta`
4. Train-val-test data splitting 
   - need a dedicated config for this part
   - the splitting is based on measurement and its corresponding dies and regions follow that splitting
   - need to add a `split` column into the meta data `df_meta` to indicates which split a row belongs to
   - the split column can take `train`, `test`, `val`, or none if the measurement is not paired
   - allow for setting up splitting ratio for train, val, and test.
   - 2 options for splitting: (1) random and (2) direct assignment
   - configurable seed for random splitting (default to None)
   - split for direct assignment takes in a dictionary with keys `train`, `val`, and `test` and their values are a list of measurement ids. Check must be made to verify one measurement only appears in one group and the 3 groups takes all paired measurements.
   - need to also output a dictionary `measurement_mapping` with keys being the measurement id from the paired source tool and the values are the corresponding paired measurement id from target tool.  
5. Data set and data loader definition
   - define dataset and data loader in this part
   - data augmentation by sliding window, or sometimes called patch, is utilized if users turn this option on in the config file 
   - sliding window should be available to each models and user can specify whether to use it or not 
   - when sliding window is on, prepare the dataset and data loader accordingly
   - details about sliding window:
      - local sequence transformation using sliding windows (patches) from the source spectrum to the target spectrum
      - window size and sliding stride can be specified in eV (default window size: 6.0 eV, default stride: 2.0 eV) or in data points
      - sliding window slices a regional spectrum into overlapping patches to capture local spectral features and augment training samples
      - right-edge anchoring guarantees 100% spectral coverage using only real physical measurements without artificial boundary padding
      - patch-to-patch mapping: predicts a target patch of length W for each input window
      - the predicted regional spectrum is reconstructed by accumulating and averaging all overlapping predicted slices across that region
      - validation loss and early stopping are evaluated directly on the reconstructed full spectrum
      - when sliding winodw is used, Bayesian optimization should allow for hyper-parameter search for window size and stride
6. Cost function defnition and preparation
   - define cost functions for model to use in this part
   - need a dedicated config for this part
   - the cost function is mainly composed of a weighted average error among all regions (default is equal weight for all region but tunable by user via the config)
   - the error in each region is defined by several weighted components (each weights are configurable ): 
      1. mean-squared error (MSE) from I_pre/max(I_target) and I_target/max(I_target), with an option to turn off the max(I_target) normalization
      2. cosine smilarity (default weight=0)
      3. mean-squared error (MSE) between gradient of predicted and true
         intensity (central difference, default weight=0)
      4. L2 regularization term (default weight=0)
   - the cost function should be able to handle patched dataset when sliding window is turn on
7. Neural network definition, training, and hyperparamter search (by Bayesian optimization)
   - need a dedicated config for this part
   - we want to train a model such that it can find the mapping of spectra from source tool to the target tool, that is, the transformation such that the measurement performed on source tool can be transformed into the measurement performed on the target tool.
   - the transformation info lies in the paired source and target measurement compiled in the "pairing source and target" step.
   - we will have many models and each of them can be specified in the config feeding to a main orchestration function
   - output should contains the trained model (best model from model training and Bayesian optimization of hyper-parameters)
   - the general architecture and workflow inside a main orchestration function will be: 
      1. model definition
      2. definition of hyper-parameter search (Bayesian optimization)
      3. model training and validation
   - here are a list of models we want to implement:
      - ResNet:
         - the model has one ResNet for each regional spectrum 
         - the ResNet is composed of 3 CNN and one shortcut (kernel size and step size are hyper-parameters)
         - padding to maintain the sequence length (by mirrow padding)
      - Residual U-Net:
         - this model combines the idea of shortcut and conventional u-net so the target y and input x has the relation "y = x + UNet(x)"
         - the model has one residual U-Net model for each regional spectrum
         - the u-net part has a default depth 3 (tunable by user)
      - U-net:
         - the same as residual u-net but now no shortcut. Namely from "y = x + UNet(x)" to just "y = UNet(x)".
         - the model has one U-Net model for each regional spectrum
8. Prediction from model
   - use the trained model from the training part to do prediction here
   - predict target spectra on train/validation/test data for model performance observation later
   - the output contains intensity, energy, and meta data
7. Model performance observation
8. Calculation of atomic percentage for elements by area integration
   - to get atomic percentage for elements you need to remove background, integrate area, and corrected with sensitivity factors
   - you do not need to deconvolute peaks, you only need to remove background signal (like using Shirley or other common functions)
   - there should be a build-in table for relative sensitivity factor (RSF) so user can modify

## Tech Stack
- Python 3
- Scikit-learn, PyTorch, Pandas, NumPy, SciPy, Ax (for Bayesian optimization)
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
   - Try to minimize coupling between modules so editing one is less likely to break the others.
   - Encapsulate internal functions that will not or should not be used by other modules.
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
10. ** Error Handling and Warning*:*
   - When error occurs provide informative details about the error.
   - Avoid silent error.
   - Always discuss with the user first about how to handle unexpected cases or error. Do not handle unexpected case silently. Do not guess what the fix should be without asking or at least notifying the user. 
11. ** Code Line Count Reduction**
   - When asked to reduce the number of code lines (comment and docstrings are not counted) for one specific file or many specified files:
      1. DO NOT EVER modify anything outside the specified files. 
      2. DO NOT package coding lines into another module as it is merely moving lines to somewhere else rather than cutting down lines. 
      3. DO NOT rename variables to shorten the line counts. Variable names should be easy to understand for reader. 
      4. DO NOT just use `;` to put different lines into one line. This is not cutting down lines.
      5. One coding line cannot exceed 120 characters. One comment or docstring line cannot exceed 80 characters.
      6. Note that comment and docstring lines are not counted. Exploit this to ensure code clarity and readability when reducing code lines.
      7. Comment lines and docstring should still be complete and comprehensive. No need to remove them unless explicitly requested.
   - Ask me anything if there is something unclear to you
   - Always Propose your plans first for me to review. Only implement the plan if I approve