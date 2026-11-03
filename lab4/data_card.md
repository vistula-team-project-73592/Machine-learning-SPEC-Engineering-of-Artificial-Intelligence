Dataset Card: UCI Adult Income (Processed Pipeline)

1. Dataset Summary
Dataset Name: UCI Adult Income (Census Income)

Source: OpenML (name='adult', version=2)

Task Type: Supervised Binary Classification

Domain: Socioeconomic / Demographic / Census

Target Variable: income (<=50K [Class 0] vs. >50K [Class 1])

Pipeline Implementation: lab04/src/pipeline.py

Output Artifacts:

Processed Dataset: lab04/results/processed_adult.csv

Transformer Pipeline: lab04/results/preprocessor.joblib

Audit Subset: lab04/results/label_audit_sample_50.csv

Metadata & Parameters: lab04/results/config.yaml

2. Dataset Structure
Dimensions & Splits
Split,Partition Criteria,Rows,% of Total,Purpose
Raw Ingested,Full extraction from OpenML,"48,842",100%,Initial source baseline
Train Set,"70% stratified, deduplicated","34,165",~70.0%,Model training and parameter fitting
Test Set,"30% stratified, pristine","14,653",~30.0%,Final out-of-sample evaluation
Audit Sample,Uniform random sample,50,<0.1%,Manual error and label auditing

(Note: 24 identical feature-label duplicate rows were purged strictly from the training split, lowering the raw 34,189 train subset to 34,165).

3. Feature Schema
The dataset includes 14 original predictor attributes, 2 engineered interaction features, 1 partition indicator, and 1 binary prediction target.

Feature Name,Data Type,Valid Bound / Range,Imputation Strategy,Normalization / Scaling
age,float64,"[17,90]",Train Median,z-score (StandardScaler)
fnlwgt,float64,≥0,Train Median,z-score (StandardScaler)
education-num,float64,"[1,16]",Train Median,z-score (StandardScaler)
capital-gain,float64,"[0,99999]",Train Median,z-score (StandardScaler)
capital-loss,float64,"[0,4356]",Train Median,z-score (StandardScaler)
hours-per-week,float64,"[1,99]",Train Median,z-score (StandardScaler)

Feature Name,Storage Type,Cardinality,Unknown / Missing Token,Encoding Method
workclass,string,9,?,Train Mode → One-Hot (handle_unknown='ignore')
education,string,16,N/A,One-Hot (handle_unknown='ignore')
marital-status,string,7,N/A,One-Hot (handle_unknown='ignore')
occupation,string,15,?,Train Mode → One-Hot (handle_unknown='ignore')
relationship,string,6,N/A,One-Hot (handle_unknown='ignore')
race,string,5,N/A,One-Hot (handle_unknown='ignore')
sex,string,2,N/A,One-Hot (handle_unknown='ignore')
native-country,string,42,?,Train Mode → One-Hot (handle_unknown='ignore')

Feature Name,Formula / Derivation,Justification
capital_net,capital-gain - capital-loss,Consolidates investment returns into a unified net signal.
hours_per_age,"hours-per-week / max(age - 16, 1)",Captures labor intensity normalized over working-age adulthood.

Pipeline Metadata Columns
is_train: Binary flag (1 = Training partition, 0 = Evaluation partition).

target: Cleaned binary integer target (0 for <=50K, 1 for >50K).

4. Data Quality & Preprocessing Protocol1. Missing Data HandlingMissing Token: Encoded as string '?' or native NaN.Prevalence: Confined to workclass (5.73%), occupation (5.75%), and native-country (1.75%).Policy: Categorical gaps are filled via modal values computed exclusively on the training partition. Numeric NaNs are imputed via training medians. No rows are dropped due to missing values.2. Outlier WinsorizationExtreme numeric values are clipped to the 1st and 99th percentiles. Percentile bounds are computed strictly on X_train and applied invariant to X_test.3. Leakage Prevention GuaranteesDataset is split into 70/30 train/test segments prior to running imputation, statistical profiling, quantile boundary estimation, or standard scaling.Scaling parameters ($\mu, \sigma$) and one-hot categories are calculated only on training instances via ColumnTransformer.fit().4. Label Integrity & Duplicate AuditContradictory Labels: Audited feature duplicates revealed 5 groups of identical demographic profiles with conflicting income labels (<=50K and >50K).Deduplication: 24 exact duplicates were removed from the training split. The test split was left completely unedited to preserve the true production evaluation distribution.


5. Baseline Performance Verification
Evaluated with a Logistic Regression classifier (max_iter=1000, random_state=42):

Configuration,Test Accuracy,Test F1 Score,Notes
Model A (Baseline Features),85.59%,0.6658,Original numerical + categorical attributes
Model B (Engineered Features),85.85%,0.6712,Added capital_net and hours_per_age

6. Known Biases & Ethical Considerations
Historical & Sociodemographic Bias: The source data reflects 1994 US demographic structures, containing systemic wage disparities correlated with race and sex. Models trained on this dataset inherit historical inequalities present in mid-1990s US labor markets.

Representational Skew: The dataset exhibits class imbalance (~75% <=50K vs. ~25% >50K) and geographic concentration (predominantly United States records within native-country).

Proxy Variables: High collinearity exists between demographic indicators and occupational access; naive deployment without fairness audits may amplify disparities.

