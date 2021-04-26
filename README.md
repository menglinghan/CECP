# Excluding Indirect Bias from the Association between Protected and Non-protected Attributes for the Fairness of Machine Learning Software


This repository stores our experimental codes for the paper “Excluding Indirect Bias from the Association between Protected and Non-protected Attributes for the Fairness of Machine Learning Software”，CECP is short for the method we propose in this paper: *Confounding Effect based Construction and Prediction*.

<br/>

## Datasets

We use 5 datasets, all of which are widely used in fairness research: **Adult, COMPAS, German Credit, Default and Heart Disease**. We provide these data sets in the "dataset" folder.

<br/>

## Codes for CECP

You can easily reproduce our method, we provide it in the CECP folder. 

The codes in the folder are named for the applicable scenarios. The Adult and COMPAS data sets include two protected attributes, so we divide them into two scenarios: Adult_sex (COMPAS_sex) and Adult_race (COMPAS_race). 

The code contains data preprocessing, our method and the calculation of indicators. You can run these codes directly to get the experimental results.

<br/>

## Baseline methods

We compare our method with three baseline methods:

**Fairway:** Proposed in the paper: *Fairway: A Way to Build Fair ML Software*. Fairway is a hybrid algorithm that combines pre-processing and in-processing methods. It trains the models separately according to the protected attributes to remove biased data points. Then it uses Flash technique for multi-objective optimization, including model performance indicators and fairness indicators.

We use the code they provided in the code repository: <https://github.com/joymallyac/Fairway>

**Reweighing:** Reweighing is a pre-processing method that calculates a weight value for each data point based on the expected probability and the observed probability, to help the unprivileged class have a greater chance of obtaining favorable prediction results.

We use python's AIF360 module to achieve it: 

```python
from aif360.algorithms.preprocessing import Reweighing
```

**Disparate Impact Remover (DIR):** This method is a pre-processing technology. Its main goal is to eliminate Disparate Impact and increase fairness between groups by modifying attribute values except the protected attributes.

We also use python's AIF360 module to achieve it: 

```python
from aif360.algorithms.preprocessing import DisparateImpactRemover
```

## Experimental settings
* When training the model, we refer to Fairway's attribute selection before constructing ML software, which removes some inappropriate attributes, e.g., 'native-country' on Adult dataset. Therefore, in RQ1 (Figure 4), the total number of attributes we refer to is **the number after deletion**.
* In Discussion 3, when we use AOD and EOD, we refer to Fairway's usage and report after **taking the absolute value**, less absolute values |AOD| and |EOD| mean more fairness.
