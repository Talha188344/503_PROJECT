
# Project Overview

## GitHub Repository
All usable code for the project is contained within the FINAL area. Visit the project's [GitHub repository](https://github.com/Talha188344/503_PROJECT/) for more details.

## Python Requirements
The following Python packages are required:
- pandas
- sklearn
- scikit-learn
- datetime
- numpy
- matplotlib

## Classifier Chains
There are four different Classifier Chain Python files, each serving a specific purpose:
- **CC_CORE.py**: This is the most foundational file, serving as the base for the other three. It creates the classifier chain, gets accuracies for all chains and One-vs-Rest (OVR), creates and judges the ensemble chain, and graphs the results. Inspired by this [scikit-learn guide](https://scikit-learn.org/stable/auto_examples/multioutput/plot_classifier_chain_yeast.html).
- **CC_VS_OVR.py**: Compares the OVR and ensemble classification methods in terms of accuracy, printing the results in the terminal.
- **CC_RUNTIME.py**: Measures the runtime of the classification chains versus the number of labels and versus the number of chains used in the ensemble, then graphs them.
- **CC_F1_SCORES.py**: Compares F1 Scores specifically and graphs them.

### Modifications
For all Python files, alterations might only be recommended for the dataset used for testing or the algorithms used (logistic regression and decision tree) by the chains and OVR. This is explained in comments near the top of the code and by relevant lines. All programs can be run directly, and graphs and data will be displayed/printed in the terminal.

All code related to classifier chains was created by Evan Lang. For inquiries or issues, please contact evanlang@bu.edu.

## Calibrated Label Ranking
All code related to Calibrated Label Ranking was created by Samarth Singh (Samarths@bu.edu) and Talha Jahangir (Talha98@bu.edu). If you have any questions or encounter any problems, please contact the authors.

### File
- **CLR.py**

### Functions
- `add_calibration_label(Y)`: Adds a calibration label to the dataset.
- `multilabel_to_calibrated_ranking(Y)`: Converts multilabel data to a calibrated ranking format.

### Models
- **Logistic Regression**: Implemented using scikit-learn's `LogisticRegression`.
- **Decision Tree**: (Optional) Uncomment the relevant lines in the script to use `DecisionTreeClassifier`.

### Outputs
The script outputs the Jaccard scores for each model and each label pair from the test data. These scores indicate how well the models are performing.

## Datasets Used
- **PC dataset**
- **OPENML: Reuters**: Taken from OPENML.
- **PC handmade Dataset**: Includes four CSV files serving as the test and training data. These must be located in the same area as the code. Comments in the code show how to use this dataset. The PC Dataset code was manually created by extracting values from user runs on [UserBenchmark](https://www.userbenchmark.com/). Labels were created in Excel using IF and AND statements to determine if performance criteria were met. Labels for upgrades were made by first determining if performance goals were met, then suggesting upgrades in an order that corresponds to how UserBenchmark weights each PC part.

### Postprocessing
You can find the postprocessing of the dataset in the equations for columns N-Q within `PC_DATASET.xlsx` within the final folder. Some equations (Met expectations equations) were unfortunately lost during the dataset's conversion to CSV and back. Dataset feature creation was performed by Samarth Singh and dataset postprocessing was performed by Evan Lang.
