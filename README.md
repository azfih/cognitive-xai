# Explainable Machine Learning for Dementia Status: Which Features Do Models Rely On?

A small research project on the OASIS-2 longitudinal dataset, studying which features machine learning models use to predict dementia status and how much they depend on a cognitive test (MMSE).

**Author:** Hifza Rafiq (BS Computer Science, COMSATS University Islamabad)
**Status:** Independent project, not peer reviewed. Not a clinical tool.

---

## Summary

- **Question.** Which features do machine learning models rely on to predict dementia status in older adults, how does performance change when the MMSE cognitive test is removed, and do the important features agree with medical findings?
- **Why it matters.** A digital cognitive-assessment tool should not only predict well, it should also rely on features that are consistent with medical knowledge. SHAP is used to check this.
- **What was done.** Three models (Logistic Regression, Random Forest, XGBoost) and a baseline were evaluated with and without MMSE, using person-level repeated cross-validation, and explained with SHAP.
- **Main findings.**
  1. Adding MMSE improved every metric for every model (ROC-AUC for Logistic Regression: 0.76 to 0.91) and MMSE dominated feature importance.
  2. Without MMSE, brain volume (nWBV) was the most important feature in both models, and lower brain volume pushed predictions toward dementia signs.
  3. The three models performed similarly (differences were within the spread), and the simple Logistic Regression did at least as well as the more complex ones.
  4. Most missed cases were at CDR 0.5 (very mild impairment), so a model like this should support clinical judgement and not replace it.
- **What it is not.** It is a small study of 150 people from one research centre, with no external validation. It does not diagnose, and SHAP explains the models, not the causes of dementia.

## Research questions

- **RQ1.** How well can different models predict dementia status from demographic and brain-volume features?
- **RQ2.** How much does performance change when MMSE is removed?
- **RQ3.** Which features do the models rely on, and do they match published medical findings?

## Data

- **OASIS-2** longitudinal dataset (Marcus et al., 2010): 373 visits from 150 people aged 60 to 96, each seen 2 to 5 times. Only the table of demographic, clinical and brain-measurement values is used, not the MRI images.
- **Label:** 1 if the Clinical Dementia Rating (CDR) is above 0, otherwise 0 (206 visits with label 0, 167 with label 1). Most label-1 visits are CDR 0.5, which is very mild impairment. CDR is never used as a feature.
- **Features:** age, sex, years of education, SES, eTIV (head size), nWBV (normalized whole-brain volume). **Experiment B** uses these; **Experiment A** adds MMSE.
- **The data are not included in this repository.** See [Data use and acknowledgement](#data-use-acknowledgement-and-citation).

## Method at a glance

- **Models:** a baseline that always predicts the most common class, Logistic Regression, Random Forest, XGBoost. Settings were fixed in advance and not tuned.
- **Testing:** the same person appears in several visits, so all visits of one person were kept together (grouped 5-fold cross-validation), repeated 10 times with different random groupings (50 test rounds). Missing values (SES 19, MMSE 2) were filled with the median using only the training part of each round.
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC (mean ± standard deviation).
- **Explainability:** SHAP on held-out people only, for Logistic Regression and Random Forest, averaged over five groupings.

## Results

Mean ± standard deviation over 50 test rounds (full table in `report/results_v2_repeated.csv`):

| Model | ROC-AUC without MMSE | ROC-AUC with MMSE | Recall without MMSE | Recall with MMSE |
|---|---|---|---|---|
| Baseline (always majority) | 0.50 | 0.50 | 0.00 | 0.00 |
| Logistic Regression | 0.76 ± 0.09 | 0.91 ± 0.04 | 0.64 ± 0.12 | 0.73 ± 0.09 |
| Random Forest | 0.69 ± 0.09 | 0.87 ± 0.04 | 0.56 ± 0.11 | 0.68 ± 0.09 |
| XGBoost | 0.69 ± 0.08 | 0.86 ± 0.04 | 0.59 ± 0.13 | 0.69 ± 0.10 |

![ROC-AUC with and without MMSE](figures/auc_with_without_mmse.png)

**What the models rely on (SHAP).** Each dot is one visit. Dots to the right of zero push the prediction toward dementia signs. Colour shows the feature value (red high, blue low).

<table>
  <tr>
    <th></th>
    <th>Logistic Regression</th>
    <th>Random Forest</th>
  </tr>
  <tr>
    <td><b>Without MMSE</b></td>
    <td><img src="figures/shap_logreg_B.png" width="420" alt="SHAP, Logistic Regression, without MMSE"></td>
    <td><img src="figures/shap_rf_B.png" width="420" alt="SHAP, Random Forest, without MMSE"></td>
  </tr>
  <tr>
    <td><b>With MMSE</b></td>
    <td><img src="figures/shap_logreg_A.png" width="420" alt="SHAP, Logistic Regression, with MMSE"></td>
    <td><img src="figures/shap_rf_A.png" width="420" alt="SHAP, Random Forest, with MMSE"></td>
  </tr>
</table>

**Reading the SHAP results.** Without MMSE, nWBV, sex and education ranked highest in both models. With MMSE, MMSE dominated, at roughly three times the importance of the next feature. Lower nWBV, fewer years of education and lower MMSE scores pushed predictions toward dementia signs, which agrees with the literature. The sex pattern reflects how this cohort was assembled and is not interpreted as a finding about sex and dementia.

## How to run

Tested with Python 3.14.

```bash
# 1. Get the code
git clone https://github.com/<your-username>/gks-cognitive-xai.git
cd gks-cognitive-xai

# 2. Create an environment and install packages
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# 3. Add the data (not included): place oasis_longitudinal.csv in data/

# 4. Start Jupyter and run the notebooks in order
jupyter notebook
```

In each notebook, use **Kernel → Restart and Run All**. Results are saved to `report/` and figures to `figures/`. SHAP values can differ very slightly between runs.

## Repository structure

```
├── data/                          not included (see Data)
├── notebooks/
│   ├── 01_data_exploration.ipynb          first look at the data
│   ├── 02_baseline_and_experiments.ipynb  baseline and Experiments A and B
│   ├── 03_repeated_validation.ipynb       10 repeats of grouped cross-validation
│   ├── 04_error_analysis.ipynb            confusion matrix, misses by CDR level
│   └── 05_shap_explainability.ipynb       SHAP for two models, rank checks
├── report/
│   ├── protocol.md                        study plan written before running models
│   ├── findings_notes.md                  working notes
│   └── results_*.csv                      saved results
├── figures/                       plots used in this README
├── requirements.txt
└── README.md
```



## Limitations

- 150 people from a single research centre, mostly volunteers, and no external validation set.
- The 50 test rounds reuse the same people, so the spread reflects how people were split and not uncertainty in a new sample.
- The label includes very mild impairment and is not a confirmed diagnosis.
- Some features are related to each other (age and nWBV, education and SES), so SHAP credit can be shared between them.
- Only three models with fixed settings were compared.

## Data use, acknowledgement and citation

The OASIS-2 data are **not redistributed** here. The data can be requested from OASIS (www.oasis-brains.org) under its Data Use Agreement. A copy of the tabular file used in this project was obtained from Kaggle, and OASIS was contacted about official access.

Marcus, D. S., Fotenos, A. F., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2010). Open Access Series of Imaging Studies: Longitudinal MRI data in nondemented and demented older adults. *Journal of Cognitive Neuroscience, 22*(12), 2677–2684. https://doi.org/10.1162/jocn.2009.21407

## Use of AI

AI assistance (Claude, Anthropic) was used as a supplementary tool for code drafting, debugging support, and language editing. The author determined the research questions and methodology, reviewed and adapted the code, independently conducted and executed the experiments, analyzed and verified the results against the dataset and cited literature, and takes full responsibility for the research and its findings.

## Contact

Hifza Rafiq, hifzarafiq19@gmail.com
