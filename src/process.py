import numpy as np
from numpy import genfromtxt
import tensorflow as tf


def read_file(unsupervised_gene, supervised_gene, gpr, pp, phenotype):
    """
    Read data files and preprocess the data.

    Args:
        unsupervised_gene (str): File path for unsupervised gene data.
        supervised_gene (str): File path for supervised gene data.
        gpr (str): File path for gene-protein mask data.
        pp (str): File path for protein-phenotype mask data.
        phenotype (str): File path for phenotype data.

    Returns:
        tuple: A tuple containing preprocessed data:
            - gene (numpy.ndarray): Preprocessed unsupervised gene data.
            - sp_gene_train (numpy.ndarray): Preprocessed supervised gene data.
            - phenotype (numpy.ndarray): Preprocessed phenotype data.
            - gene_protein_mask_tensor (tensorflow.Tensor): Gene-protein mask as a TensorFlow constant.
            - protein_pheno_mask_tensor (tensorflow.Tensor): Protein-phenotype mask as a TensorFlow constant.
            - re_gene_protein_mask_tensor (tensorflow.Tensor): Transposed gene-protein mask as a TensorFlow constant.
            - re_protein_pheno_mask_tensor (tensorflow.Tensor): Transposed protein-phenotype mask as a TensorFlow constant.

    """

    # Read and preprocess unsupervised gene data
    gene_train_raw = genfromtxt(unsupervised_gene, delimiter=',', dtype="float32")
    gene_train_raw = np.maximum(np.minimum(gene_train_raw, 10), -10) / 10
    gene = genfromtxt(unsupervised_gene, delimiter=',', dtype="float32")

    # Read and preprocess supervised gene data
    sp_gene_train = genfromtxt(supervised_gene, delimiter=',', dtype="float32")
    sp_gene_train = np.maximum(np.minimum(sp_gene_train, 10), -10) / 10

    # Read phenotype data
    phenotype = genfromtxt(phenotype, delimiter=',', dtype="float32")

    # Read and process gene-protein mask data
    gene_protein_mask = genfromtxt(gpr, delimiter=',', dtype="float32")[1:, 1:]
    gene_protein_mask_tensor = tf.constant(gene_protein_mask)
    re_gene_protein_mask_tensor = tf.transpose(gene_protein_mask_tensor)

    # Read and process protein-phenotype mask data
    protein_phenotype_mask = genfromtxt(pp, delimiter=',', dtype="float32")[1:, 1:].transpose()
    protein_pheno_mask_tensor = tf.constant(protein_phenotype_mask)
    re_protein_pheno_mask_tensor = tf.transpose(protein_pheno_mask_tensor)

    return (gene, sp_gene_train, phenotype, gene_protein_mask_tensor, protein_pheno_mask_tensor, \
            re_gene_protein_mask_tensor, re_protein_pheno_mask_tensor)
