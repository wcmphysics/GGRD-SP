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
   - meta data (`df_meta`) is a pandas dataframe containing all meta data of the spectra recorded 
   - meta data includes: material, tool and time of measurement, die, region, number of points in one spectrum, and the corresponding index in the intensity and energy array
2. Pairing source and target spectra
   - allow users to specify source tool and target tool
   - for every measurement in the source tool, we find/map a measurement from the target tool
   - this source to target mapping must be 1-to-1 (no multiple source measurement is linked to the same target measurement)
   - the mapping maps measurements from the source tool to the target tool, the die and region in that measurement will also be paired accodringly (spectrum from die A region B of the source measurement will map to the same die A and the same region B in the target measurement) 
   - the time difference between source and target measurement should be lower than a user-specified threshold (default to 12 hours) 
   - store this pairing information in the meta data dataframe (add columns like tool_target, measurement_id_target, spectrum_index_target)
3. Neural network definition, training, and hyperparamter search (by Bayesian optimization)
   - we want to train a model such that it can find the mapping of spectra from source tool to the target tool, that is, the transformation such that the measurement performed on source tool can be transformed into the measurement performed on the target tool.
   - the transformation info lies in the paired source and target measurement compiled in the "pairing source and target" step, note that material, 
   - we will have many models and each of them should be called independently via a main orchestration function
   - we want to apply data augmentation by sliding window to each models and user can specify whether to use this augmentation or not 
   - the general architecture and workflow inside a main orchestration function will be: 
      1. train-val-test data splitting (based on measurement and the splitting applies to all regions) 
      2. sliding window definition/preparation if needed
      3. dataset and data loader definition/preparation
      4. cost function definition/preparation
      5. model definition
      6. definition of hyper-parameter search (Bayesian optimization)
      7. model training and validation
      8. after training completes, apply the trained model to predict target spectra for train/validation/test data (for model performance observation later), the output is a standardized output with corresponding energy/intensity/meta data output
   - details about sliding window:
      - local sequence transformation using sliding windows (patches) from the source spectrum to the target spectrum
      - window size and sliding stride can be specified in eV (default window size: 2.0 eV, default stride: 1.0 eV) or in data points
      - pure unpadded sliding window slices a regional spectrum into overlapping patches to capture local spectral features and augment training samples
      - right-edge anchoring guarantees 100% spectral coverage using only real physical measurements without artificial boundary padding
      - patch-to-patch mapping: predicts a target patch of length W for each input window
      - the predicted regional spectrum is reconstructed by accumulating and averaging all overlapping predicted slices across that region
      - validation loss and early stopping are evaluated directly on the reconstructed full spectrum
      - when sliding winodw is used, Bayesian optimization (searching window size, kernel size, learning rate, L2 regularization)
   - definition of the cost function:
      - the cost function is mainly composed of a weighted average error among all regions (default is equal weight for all region but tunable by user via config)
      - the error in each region is calculated by first normalize intensity of true and predicted spectrum by the maximun intensity of true spectrum, and then calculate mean-squared error (MSE). This MSE is the error of this region
      - the cost function also contains a L2 regularization term (with user-tunable weight)
   - here are a list of models we want to implement:
   1. ResNet:
      - the model is many ResNet (each for one regional spectrum) 
      - the ResNet is composed of 3 CNN and one shortcut (kernel size and step size are hyper-parameters)
      - padding to maintain the sequence length (by mirrow padding)
   2. residual U-Net:
      - this model combines the idea of shortcut and conventional u-net so the target y and input x has the relation "y = x + UNet(x)"
      - the u-net part has a default depth 3 (tunable by user)
   3. u-net:
      - the same as residual u-net but now no shortcut. Namely from "y = x + UNet(x)" to just "y = UNet(x)".
4. Model performance observation
5. Calculation of atomic percentage for elements by area integration
   - to get atomic percentage for elements you need to remove background, integrate area, and corrected with sensitivity factors
   - you do not need to deconvolute peaks, you only need to remove background signal (like using Shirley or other common functions)
   - there should be a build-in table for relative sensitivity factor (RSF) so user can modify

## Tech Stack
- Python 3
- Scikit-learn, PyTorch, Pandas, NumPy, SciPy, Ax (for Bayesian optimization)
- Jupyter Notebooks for exploration, Python scripts for utilities.

## Agent Instructions (Rules for Gemini)
0. Follow ASD-STE100 Simplified Technical English rule closely in communication with the user. If communication is not conducted in Chinese, follow these rules: 
   - 限制字數在30字以內
   - 少用或不用被動句(如"被"、"讓")
   - 禁止省略主詞
   - 一句話只包含一個核心動作，禁止使用連續動詞句
   - 句子中的條件必須要前置(比如先寫"如果..."，再接著寫"就...")
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