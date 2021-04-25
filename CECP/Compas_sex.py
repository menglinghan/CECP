import pandas as pd
import random,time,csv
import numpy as np
import math,copy,os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from aif360.sklearn import metrics
import sys
sys.path.append(os.path.abspath('..'))
np.random.seed(10)

## Load dataset
dataset_orig = pd.read_csv('../dataset/compas-scores-two-years.csv')

## Drop categorical features
dataset_orig = dataset_orig.drop(['id','name','first','last','compas_screening_date','dob','age','juv_fel_count','decile_score','juv_misd_count','juv_other_count','days_b_screening_arrest','c_jail_in','c_jail_out','c_case_number','c_offense_date','c_arrest_date','c_days_from_compas','c_charge_desc','is_recid','r_case_number','r_charge_degree','r_days_from_arrest','r_offense_date','r_charge_desc','r_jail_in','r_jail_out','violent_recid','is_violent_recid','vr_case_number','vr_charge_degree','vr_offense_date','vr_charge_desc','type_of_assessment','decile_score','score_text','screening_date','v_type_of_assessment','v_decile_score','v_score_text','v_screening_date','in_custody','out_custody','start','end','event'],axis=1)

## Drop NULL values
dataset_orig = dataset_orig.dropna()

## Change symbolics to numerics
dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'Female', 1, 0)
dataset_orig['race'] = np.where(dataset_orig['race'] != 'Caucasian', 0, 1)
dataset_orig['priors_count'] = np.where((dataset_orig['priors_count'] >= 1 ) & (dataset_orig['priors_count'] <= 3), 3, dataset_orig['priors_count'])
dataset_orig['priors_count'] = np.where(dataset_orig['priors_count'] > 3, 4, dataset_orig['priors_count'])
dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Greater than 45',45,dataset_orig['age_cat'])
dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == '25 - 45', 25, dataset_orig['age_cat'])
dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Less than 25', 0, dataset_orig['age_cat'])
dataset_orig['c_charge_degree'] = np.where(dataset_orig['c_charge_degree'] == 'F', 1, 0)

## Rename class column
dataset_orig.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)

from sklearn.preprocessing import MinMaxScaler
from scipy import stats

scaler = MinMaxScaler()
dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)

aod_b = []
eod_b = []
aod_a = []
eod_a = []
acc_a = []
acc_b = []
ce_list = []
recall_a = []
recall_b = []
false_a = []
false_b = []
TI_a = []
DI_a = []
SPD_a = []
TI_b = []
DI_b = []
SPD_b = []
ce_times = []

for i in range(100):
    print('------the {}th turn------'.format(i))

    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.15, random_state=None, shuffle=True)

    X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
    X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

    column_train = [column for column in X_train]

    clf = LogisticRegression()

    from aif360.datasets import BinaryLabelDataset, StructuredDataset
    from aif360.algorithms.preprocessing import Reweighing
    from aif360.metrics import ClassificationMetric
    from aif360.metrics import BinaryLabelDatasetMetric

    dataset_t = BinaryLabelDataset(favorable_label=1.0,
                                   unfavorable_label=0.0,
                                   df=dataset_orig_test,
                                   label_names=['Probability'],
                                   protected_attribute_names=['sex'],
                                   )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    dataset_pred = dataset_t.copy()
    dataset_pred.labels = y_pred
    attr = dataset_t.protected_attribute_names[0]
    idx = dataset_t.protected_attribute_names.index(attr)
    privileged_groups = [{attr: dataset_pred.privileged_protected_attributes[idx][0]}]
    unprivileged_groups = [{attr: dataset_pred.unprivileged_protected_attributes[idx][0]}]

    class_metrics = ClassificationMetric(dataset_t, dataset_pred, unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)
    b_metrics = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)
    print("theil_index:", class_metrics.theil_index())
    print("Disparate Impact:", class_metrics.disparate_impact())
    print("Statistical Parity Difference:", class_metrics.statistical_parity_difference())
    TI_b.append(class_metrics.theil_index())
    DI_b.append(class_metrics.disparate_impact())
    SPD_b.append(class_metrics.statistical_parity_difference())

    print("accuracy :", class_metrics.accuracy())
    print("recall :", class_metrics.recall())
    print("far:", class_metrics.false_positive_rate())
    print("aod:", class_metrics.average_odds_difference())
    print("eod:", class_metrics.equal_opportunity_difference())
    acc_b.append(class_metrics.accuracy())
    recall_b.append(class_metrics.recall())
    false_b.append(class_metrics.false_positive_rate())
    aod_b.append(class_metrics.average_odds_difference())
    eod_b.append(class_metrics.equal_opportunity_difference())

    slope_store = []
    intercept_store = []
    rvalue_store = []
    pvalue_store = []
    column_u = []
    flag = 0
    ce = []
    times = 0
    def Linear_regression(x, slope, intercept):
        return x * slope + intercept
    for i in column_train:
        flag = flag + 1
        if i != 'sex':
            slope,intercept,rvalue,pvalue,stderr=stats.linregress(X_train['sex'], X_train[i])
            rvalue_store.append(rvalue)
            pvalue_store.append(pvalue)
            if i != 'race':
                if pvalue < 0.05:
                    times = times + 1
                    column_u.append(i)
                    ce.append(flag)
                    slope_store.append(slope)
                    intercept_store.append(intercept)
                    X_train[i] = X_train[i] - Linear_regression(X_train['sex'], slope, intercept)

    ce_times.append(times)
    ce_list.append(ce)

    X_train = X_train.drop(['sex'],axis = 1)

    for i in range(len(column_u)):
        X_test[column_u[i]] = X_test[column_u[i]] - Linear_regression(X_test['sex'], slope_store[i], intercept_store[i])


    X_test = X_test.drop(['sex'],axis = 1)

    from aif360.datasets import BinaryLabelDataset, StructuredDataset
    from aif360.algorithms.preprocessing import Reweighing
    from aif360.metrics import ClassificationMetric
    from aif360.metrics import BinaryLabelDatasetMetric

    dataset_t = BinaryLabelDataset(favorable_label=1.0,
                                   unfavorable_label=0.0,
                                   df=dataset_orig_test,
                                   label_names=['Probability'],
                                   protected_attribute_names=['sex'],
                                   )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    dataset_pred = dataset_t.copy()
    dataset_pred.labels = y_pred
    attr = dataset_t.protected_attribute_names[0]
    idx = dataset_t.protected_attribute_names.index(attr)
    privileged_groups = [{attr: dataset_pred.privileged_protected_attributes[idx][0]}]
    unprivileged_groups = [{attr: dataset_pred.unprivileged_protected_attributes[idx][0]}]

    class_metrics = ClassificationMetric(dataset_t, dataset_pred, unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)
    b_metrics = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)
    print("theil_index:", class_metrics.theil_index())
    print("Disparate Impact:", class_metrics.disparate_impact())
    print("Statistical Parity Difference:", class_metrics.statistical_parity_difference())
    TI_a.append(class_metrics.theil_index())
    DI_a.append(class_metrics.disparate_impact())
    SPD_a.append(class_metrics.statistical_parity_difference())

    print("accuracy :", class_metrics.accuracy())
    print("recall :", class_metrics.recall())
    print("far:", class_metrics.false_positive_rate())
    print("aod:", class_metrics.average_odds_difference())
    print("eod:", class_metrics.equal_opportunity_difference())
    acc_a.append(class_metrics.accuracy())
    recall_a.append(class_metrics.recall())
    false_a.append(class_metrics.false_positive_rate())
    aod_a.append(class_metrics.average_odds_difference())
    eod_a.append(class_metrics.equal_opportunity_difference())

#DI,SPD are the original value, different from the value reported in the paper.
print('---Original---')
print('Aod before:', np.mean(np.abs(aod_b)))
print('Eod before:', np.mean(np.abs(eod_b)))
print('Acc before:', np.mean(acc_b))
print('Far before:', np.mean(false_b))
print('recall before:', np.mean(recall_b))
print('TI before:', np.mean(TI_b))
print('DI before:', np.mean(DI_b))
print('SPD before:', np.mean(np.abs(SPD_b)))

print('---CECP---')
print('Aod after:', np.mean(np.abs(aod_a)))
print('Eod after:', np.mean(np.abs(eod_a)))
print('Acc after:', np.mean(acc_a))
print('Far after:', np.nanmean(false_a))
print('recall after:', np.nanmean(recall_a))
print('TI after:', np.mean(TI_a))
print('DI after:', np.mean(DI_a))
print('SPD after:', np.mean(np.abs(SPD_a)))



