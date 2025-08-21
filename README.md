Credit Card Fraud Detection with Generative Models

This repository contains all of the code, experiments and results for my MSc project "Credit Card Fraud Detection using Generative Models". The goal of the project is to explore how modern machine‑learning and deep‑learning techniques—together with generative models that create synthetic fraud examples—can improve detection performance on highly imbalanced transaction data. The work spans classical supervised learning, unsupervised anomaly detection, deep sequential models and several variants of Generative Adversarial Networks (GANs).

Project structure
├── src/                 # Python scripts for training models and running experiments
├── dataset/             # Local copy of the Kaggle credit‑card fraud dataset (not committed to GitHub)
├── cgan.jpg             # PCA projection of CGAN‑generated fraud vs. real fraud
├── ctgan.jpg            # PCA projection of CTGAN‑generated fraud vs. real fraud
├── gan.jpg              # PCA projection of vanilla GAN‑generated fraud vs. real fraud
├── vaegan.png           # PCA projection of VAE‑GAN‑generated fraud vs. real fraud
├── wcgan.jpg            # PCA projection of Wasserstein conditional GAN‑generated fraud vs. real fraud
├── wgan.jpg             # PCA projection of Wasserstein GAN‑generated fraud vs. real fraud
├── methodology.png      # High‑level flowchart of the experimental methodology
└── README.md            # You are here


Note: The raw dataset is intentionally excluded from version control because it is over 100 MB. Download the data yourself (see installation instructions below) and place it in the dataset/ directory.

Installation

These instructions assume a Unix‑like environment and Python 3.8+. Adjust the commands for your operating system where needed.

Clone the repository

git clone https://github.com/aditi2306/2425-CT5129-fraud-detection-code.git
cd 2425-CT5129-fraud-detection-code


Set up a virtual environment (optional but recommended)

python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate


Install Python dependencies

This project uses a variety of packages—scikit‑learn, imbalanced‑learn, LightGBM, XGBoost, PyTorch/TensorFlow, etc.—to train classical models, oversamplers and generative models. A requirements.txt file is provided in src/ listing the exact versions used for the experiments. Install the requirements with:

pip install -r src/requirements.txt


Download the dataset

The experiments are based on the publicly available Credit Card Fraud Detection dataset originally released on Kaggle by the ULB Machine Learning Group
raw.githubusercontent.com
. Download the CSV file from Kaggle and save it as dataset/creditcard.csv in the project root. Do not commit the dataset to GitHub (it exceeds the 100 MB file size limit and is tracked through Git LFS locally).

Run an experiment

All training and evaluation scripts live in the src/ folder. For example, to train an XGBoost model on the original imbalanced data run:

python src/train_xgboost.py --input dataset/creditcard.csv --model_out models/xgb.pkl


To train a GAN and then a classifier on GAN‑augmented data, use one of the GAN scripts (e.g. train_ctgan.py) followed by the classifier script (e.g. train_rf_on_synth.py). See the docstrings inside each script for configurable arguments such as the number of training epochs, the size of the synthetic data set and the choice of downstream classifier.

Methodology

The overall experimental pipeline is summarised in the following flowchart:

Data loading and preprocessing – The anonymised credit‑card transactions from the Kaggle dataset are loaded. Features are scaled with a StandardScaler to zero mean and unit variance. The dataset is highly imbalanced (≈0.17 % fraud), so a train/validation/test split preserves class proportions.

Baseline supervised models – We first train several conventional classifiers on the raw imbalanced data: Logistic Regression, Support Vector Machine (SVM), Random Forest, XGBoost and LightGBM. Their ROC‑AUC scores exceed 0.95 but the PR‑AUC and F1 scores reveal poor precision–recall balance, motivating the need for resampling or generative approaches.

Resampling strategies – We experiment with the Synthetic Minority Over‑sampling Technique (SMOTE), Borderline‑SMOTE and ADASYN. These methods create synthetic minority samples before training a classifier. LightGBM with Borderline‑SMOTE achieved a PR‑AUC of ≈0.85 and F1 of ≈0.84, improving recall at the expense of more false positives.

Unsupervised anomaly detection – Isolation Forest and One‑Class SVM serve as unsupervised baselines. They yield respectable ROC‑AUC scores (~0.95) but much lower PR‑AUC (0.14 and 0.35 respectively) and F1 scores (≈0.25–0.47), underlining the difficulty of detecting rare frauds without labels.

Autoencoder – A neural autoencoder is trained to reconstruct normal transactions. Reconstruction errors are used as anomaly scores. With an appropriately tuned threshold, the autoencoder achieved ROC‑AUC ≈0.977 and PR‑AUC ≈0.79, outperforming the unsupervised baselines.

GAN‑based oversampling – We implement and compare several GAN variants to generate synthetic frauds:

GAN – vanilla GAN trained with a multilayer perceptron generator and discriminator. Synthetic fraud samples are highly variable but often unrealistic, leading to moderate PR‑AUC improvements when used to train a classifier.

WGAN – Wasserstein GAN with gradient penalty. Produces more stable samples but can suffer from mode collapse. When paired with LightGBM or Random Forest it offers good recall but limited precision.

WCGAN – Wasserstein conditional GAN, conditioning on the fraud label. It improves sample quality but still shows instability in precision–recall trade‑off.

CGAN – Conditional GAN that incorporates class information in both generator and discriminator. The synthetic fraud distribution better matches real fraud patterns.

CTGAN – Conditional Tabular GAN designed for tabular data. It uses Gaussian mixture conditioning to model discrete and continuous features. CTGAN yielded the best trade‑off, with Random Forest trained on CTGAN data achieving PR‑AUC ≈0.888 and F1 ≈0.87.

VAE‑GAN – Combines a variational autoencoder (VAE) with a GAN objective. It produces diverse yet realistic fraud samples. VAE‑GAN with Random Forest achieved PR‑AUC ≈0.82 and F1 ≈0.83.

Each GAN is trained on between 5 % and 100 % of the fraud training set to investigate how sample size influences generation quality. The resulting synthetic data are combined with real training data to train classifiers such as Random Forest, LightGBM, LSTM and GRU networks.

Sequential models – To capture temporal patterns in transaction sequences we train recurrent networks (RNN, LSTM, GRU) on transaction histories. GAN‑ and CTGAN‑augmented LSTM models achieved high ROC‑AUC (~0.97–0.99) and PR‑AUC (~0.85) with F1 around 0.84, outperforming purely sequential models on raw data.

Evaluation – We evaluate models on a held‑out test set using multiple metrics: accuracy, precision, recall, F1‑score, ROC‑AUC and Precision–Recall AUC (PR‑AUC). Confusion matrices provide insights into false positives and false negatives. The PR‑AUC and F1‑score are emphasised because they better reflect minority class performance.

Results overview
Synthetic data quality

The scatter plots below visualise real fraud samples (blue) versus fraud samples generated by each GAN variant (orange) after projecting the data down to two principal components. These figures help illustrate how closely each model’s output matches the true fraud distribution.

Model	PCA projection of real (blue) vs. synthetic fraud (orange)
GAN	

WGAN	

WCGAN	

CGAN	

CTGAN	

VAE‑GAN	

The CTGAN and VAE‑GAN plots show that synthetic fraud points (orange) cluster closely around real fraud regions, indicating that these models best capture the underlying fraud patterns. In contrast, the vanilla GAN and WGAN models produce more scattered or collapsed distributions, which explains their weaker precision–recall performance.

Performance summary

The table below summarises the best metrics achieved by each modelling approach. Higher values are better.

Method / configuration	ROC‑AUC	PR‑AUC	F1‑score (fraud)	Notes
Logistic Regression	0.997	0.691	0.54	High ROC‑AUC but poor recall; baseline indicator
SVM (RBF)	0.977	0.435	0.47	Struggles with extreme imbalance
Random Forest	0.975	0.815	0.75	Good off‑the‑shelf performer
XGBoost	0.997	0.859	0.84	Best among classical models
LightGBM (SMOTE)	0.969	0.849	0.78	Resampling improves recall
Autoencoder	0.977	0.791	0.79	Strong unsupervised baseline
Isolation Forest	0.954	0.137	0.25	Poor precision–recall
One‑Class SVM	0.952	0.354	0.47	Better than Isolation Forest
CTGAN + Random Forest	0.965	0.888	0.87	Best overall trade‑off; top PR‑AUC and F1
CGAN + LightGBM	0.977	0.848	0.86	Strong combination
WGAN + LightGBM	0.989	0.802	0.83	High precision but lower recall
VAE‑GAN + Random Forest	0.973	0.820	0.83	Balanced augmentation
GAN + LSTM	0.993	0.861	0.84	Top sequential performer
CTGAN + LSTM	0.971	0.847	0.83	Good recall and precision
WGAN + LSTM	0.981	0.864	0.84	Strong sequential model
WCGAN + LSTM	0.957	0.741	0.78	High precision but lower recall

Across all experiments, augmenting the minority class with synthetic fraud samples generated by CTGAN and VAE‑GAN yields the largest improvements in PR‑AUC and F1‑score, while maintaining high ROC‑AUC. Sequential models (e.g. LSTM) further capture temporal patterns and, when coupled with GAN augmentation, deliver the best overall performance.

Contributing

Contributions and suggestions are welcome. Feel free to fork the repository, open issues or pull requests. Please ensure that any added datasets remain under the 100 MB GitHub limit or are stored via Git LFS.

License

This project is released under the MIT license. See the LICENSE
 file for details.
