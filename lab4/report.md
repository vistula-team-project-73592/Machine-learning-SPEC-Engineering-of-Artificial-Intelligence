Lab 04 Report: Census Data Preprocessing and Leakage-Free Pipeline Architecture

1. Problem Formulation
This laboratory focuses on binary income classification using the Adult Census dataset, predicting whether an individual's annual earnings exceed $50,000 (>50K) or remain below or equal to that threshold (<=50K) across 14 demographic and occupational attributes.

Target Variable: Binary annual income tier (<=50K vs. >50K).

Evaluation Criteria: Classification Accuracy and F1 Score. The F1 metric is critical due to severe class imbalance, as roughly 75% of observations belong to the <=50K majority class. A naive majority-class baseline would yield an inflated 75% accuracy while offering zero diagnostic utility for high-income earners.

Core Engineering Constraints: Complete prevention of data leakage through strict separation of training and test distributions, coupled with fully automated, single-command pipeline reproducibility.


2. Ingestion and Exploratory Data ValidationRaw Data ExtractionThe dataset was ingested directly from OpenML without preliminary transformations:Total Sample Count: 48,842 observationsDimensionality: 15 variables (14 predictor features and 1 target label)Range Integrity and Boundary VerificationContinuous variables were audited against biologically and logically valid parameter ranges:age ($[17, 90]$): 0 out-of-bounds violations.hours-per-week ($[1, 99]$): 0 out-of-bounds violations.capital-gain ($[0, 99999]$): 0 out-of-bounds violations.capital-loss ($[0, 4356]$): 0 out-of-bounds violations.Missing Value ProfilingMissing observations encoded as question marks (?) were isolated to three categorical features:workclass: 2,799 unrecorded entries (5.73%)occupation: 2,809 unrecorded entries (5.75%)native-country: 857 unrecorded entries (1.75%)All observations were preserved, and missingness metrics were explicitly logged rather than dropped via row-wise deletion.



3. Label Inconsistency Audit
Contradictory Duplicates: Identified 5 clusters of identical feature profiles with conflicting class labels (one instance labeled <=50K alongside an exact duplicate labeled >50K). This underscores intrinsic label noise and unobserved latent confounders characteristic of real-world survey methodology.

Audit Verification: Exported a 50-record stratified random subset to results/label_audit_sample_50.csv for qualitative validation.



4. Leakage-Free Preprocessing Pipeline
To guarantee rigorous evaluation without data leakage:

Partition Precedence: The raw dataset was partitioned prior to any transformation into a 70% training split (34,189 samples) and a 30% evaluation split (14,653 samples).

Distributional Isolation: Imputation statistics were calculated strictly on the training partition (medians for continuous predictors and modes for categorical variables) and subsequently applied to transform test samples out-of-sample.

Outlier Mitigation: Winsorization bounds were established using the 1st and 99th percentiles derived exclusively from the training split.

Deduplication: Removed 24 exact duplicate rows from the training subset while leaving the evaluation partition completely unperturbed.

Artifact Serialization: Exported the fitted transformation state to results/preprocessor.joblib.


5. Feature Engineering and Baseline Performance
Two composite domain features were engineered:

capital_net = capital-gain - capital-loss (net realized investment capital).

hours_per_age = hours-per-week / (age - 16) (work intensity normalized by career-eligible adult lifespan).
Model Configuration,Classification Accuracy,Macro/Weighted F1 Score
Baseline Features (Raw Pipeline),85.59%,0.6658
Augmented Features (Engineered Pipeline),85.85%,0.6712

Performance gains across both metrics confirm that consolidated net capital yields a clearer financial discriminant than separated gains and losses, while labor intensity relative to age helps distinguish high-earning career trajectories.

6. Execution and Verification
End-to-End Execution Command: Executable via python lab04\src\pipeline.py.

Dataset Artifact Integrity (SHA-256):
16c15d72d4ab8d7453a884edb90dccbdb8344d9da918ed9d18359b6c8490a04d

Configuration & Audit Trace: Hyperparameters, pipeline metadata, and validation outputs are tracked in results/config.yaml.