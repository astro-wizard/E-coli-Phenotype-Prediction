# E-coli Phenotype Prediction (Biological inspired neural network)
This repository contains code for a Biological Inspired Neural Network (BI-NN) that predicts the phenotype of E. coli bacteria based on their transcriptomic profile of gene expressions. The BI-NN model utilizes an autoencoder architecture along with gene-protein and protein-phenotype masks to learn the underlying patterns in the data and make predictions.

## Installation
To run the code in this repository, please follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-username/repo-name.git
```
2. Navigate to the project directory:
```bash
cd repo-name
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

2. Run the pipeline:
```bash

```

