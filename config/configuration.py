from configparser import ConfigParser

config = ConfigParser()

config["DATA"] = {
    "supervised_gene": "../data/raw/supervised_transcriptomics.csv",
    "unsupervised_gene": "../data/raw/unsupervised_transcriptomics.csv",
    "gene_protein": "../data/association_rule/gene_protein_rule.csv",
    "protein_phenotype": "../data/association_rule/protein_phenotype_rule.csv",
    "phenotype": "../data/raw/supervised_phenotype.csv"
}
config["MODEL"] = {
    "autoencoder model": "../model/autoencoder_model.h5",
    "Supervised model": "../model/supervised_model.h5"
}
config["PREDICT"] = {
    "predict_supervised": "../data/toy/toy_supervised_transcriptomics.csv",
    "prediction": "..data/toy/supervised_prediction.csv"
}

with open("config.yaml", "w") as f:
    config.write(f)
