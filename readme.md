# E-coli Phenotype Prediction (Biological inspired neural network)
This repository contains code for a Biological Inspired Neural Network (BI-NN) that predicts the phenotype of E. coli bacteria based on their transcriptomic profile of gene expressions. The BI-NN model utilizes an autoencoder architecture along with gene-protein and protein-phenotype masks to learn the underlying patterns in the data and make predictions.



## Installation
To run the code in this repository, please follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/prabhattiwari16/E-coli-Phenotype-Prediction.git
```
2. Navigate to the project directory:
```bash
cd E-coli-Phenotype-Prediction # path to the repository
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Update the configuration file:

Edit the `config/config.py` file to specify the file paths for the input data and models. Modify the following parameters according to your data:

- `unsupervised_gene`: Path to the unsupervised gene expression data file.
- `supervised_gene`: Path to the supervised gene expression data file.
- `phenotype`: Path to the phenotype data file.
- `gene_protein`: Path to the gene-protein mask file.
- `protein_phenotype`: Path to the protein-phenotype mask file.
- `autoencoder_model`: Path to save the trained autoencoder model.
- `supervised_model`: Path to save the trained supervised model.
- `predict_supervised`: Path to the supervised transcriptomics data file to predict phenotypes for.
- `prediction`: Path to save the prediction results.

2. Run config.py to set up the configuration and create config.yaml file:
```bash
python config/config.py 
```
3. Once the configuration is set, run main.py to train the biological-inspired neural network:
```bash
python src/main.py
```
This script will read the data from the specified files, build the model, train the autoencoder on the unsupervised gene expression data, and then train the supervised model using the encoder layers and supervised gene expression data.

4. After training or without training(using trained model in repository), you can make predictions for E. coli phenotype using predict.py:
```bash 
python src/predict.py
```
This script will load the trained model and use it to predict the phenotype based on the provided gene expression data.
## Acknowledgments

- The biological-inspired neural network architecture was developed based on prior research and publications in the field.
- The implementation of the autoencoder and supervised models leverages the TensorFlow framework.
- The project structure and configuration setup were inspired by best practices in software engineering.

Link to download data
https://www.dropbox.com/s/7kaw8m7ozp3liyc/iJO1366_demo.tar.gz?dl=0

