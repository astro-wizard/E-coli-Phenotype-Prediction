from autoencoder import build_model
from process import read_file
from configparser import ConfigParser
from train_autoencoder import train_autoencoder
from supervised import load_model, train_supervised

config = ConfigParser()
config.read("../config/config.yaml")
config_data = config["DATA"]
config_model = config["MODEL"]

# Read file paths from the config file
unsupervised_gene = config_data["unsupervised_gene"]
supervised_gene = config_data["supervised_gene"]
phenotype = config_data["phenotype"]
gpr = config_data["gene_protein"]
pp = config_data["protein_phenotype"]
path_to_save = config_model["autoencoder model"]
path_to_save_supervised = config_model["Supervised model"]

if __name__ == '__main__':
    print("Reading file")
    # Read data from files
    unsup_gene, sup_gene, pheno, gp, pp, re_gp, re_pp = read_file(unsupervised_gene, supervised_gene, gpr, pp,
                                                                  phenotype)
    print("Reading file successful")
    print("Building model")
    # Build the autoencoder model using the read gene-protein and protein-phenotype masks
    model = build_model(gp_matrix=gp, pp_matrix=pp, t_gp_matrix=re_gp, t_pp_matrix=re_pp,
                        input_shape=unsup_gene.shape[1])
    print("Building Model successful")

    # Train the autoencoder model on the unsupervised gene expression data
    train_autoencoder(model, unsup_gene, path_to_save)
    print("Training Successful")

    # Load the trained autoencoder model
    model = load_model(path_to_save)

    print("Training Supervised")
    # Train the supervised model using the autoencoder encoder layers and supervised gene expression data
    train_supervised(model, sup_gene, pheno, path_to_save_supervised)
