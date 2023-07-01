import tensorflow as tf


class ProteinLayer(tf.keras.layers.Layer):

    """
        Custom layer for protein encoding.

        This layer takes an input tensor and applies protein encoding by performing matrix multiplication
        with the given gene-protein matrix and a learnable kernel.

        Parameters:
            gp_matrix (numpy.ndarray): Matrix for encoding gene-protein relationships.

        Returns:
            Tensor: Encoded protein representation.
    """

    def __init__(self, gp_matrix, **kwargs):
        super(ProteinLayer, self).__init__(**kwargs)
        self.gp_matrix = tf.constant(
            gp_matrix,
            dtype=tf.float32
        )

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(1366, 1366),
            initializer=tf.keras.initializers.truncated_normal(stddev=0.01),
            trainable=True)

        super(ProteinLayer, self).build(input_shape)

    def call(self, inputs):
        protein_encode = tf.matmul(
            inputs,
            tf.multiply(self.gp_matrix,
                        self.kernel))
        return protein_encode

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'gp_matrix': self.gp_matrix.numpy().tolist()
        })
        return config


class PhenoLayer(tf.keras.layers.Layer):
    """
        Custom layer for phenotype encoding.

        This layer takes an input tensor and applies phenotype encoding by performing matrix multiplication
        with the given protein-phenotype matrix and a learnable kernel, followed by adding a learnable bias term
        and applying the softplus activation function.

        Parameters:
            pp_matrix (numpy.ndarray): Matrix for encoding protein-phenotype relationships.

        Returns:
            Tensor: Encoded phenotype representation.
    """

    def __init__(self, pp_matrix, **kwargs):
        super(PhenoLayer, self).__init__(**kwargs)
        self.pp_matrix = tf.constant(
            pp_matrix,
            dtype=tf.float32)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(1366, 110),
            initializer=tf.keras.initializers.truncated_normal(stddev=0.01),
            trainable=True)

        self.bias = self.add_weight(
            name='bias',
            shape=(self.pp_matrix.shape[1],),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True)

        super(PhenoLayer, self).build(input_shape)

    def call(self, inputs):
        pheno_encode = tf.add(
            tf.matmul(inputs,
                      tf.multiply(self.pp_matrix,
                                  self.kernel)),
            self.bias)

        pheno_encode = tf.keras.activations.softplus(pheno_encode)
        return pheno_encode

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pp_matrix': self.pp_matrix.numpy().tolist()
        })
        return config


class ReconstructedProteinLayer(tf.keras.layers.Layer):

    """
    Custom layer for reconstructed protein decoding.

    This layer takes an input tensor and applies reconstructed protein decoding by performing matrix multiplication
    with the given phenotype-protein matrix and a learnable kernel, followed by adding a learnable bias term
    and applying the softplus activation function.

    Parameters:
        pp_matrix (numpy.ndarray): Matrix for decoding phenotype-protein relationships.

    Returns:
        Tensor: Decoded protein representation.
    """

    def __init__(self, pp_matrix, **kwargs):
        super(ReconstructedProteinLayer, self).__init__(**kwargs)
        self.pp_matrix = tf.constant(
            pp_matrix, dtype=tf.float32)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(110, 1366),
                                      initializer=tf.keras.initializers.truncated_normal(stddev=0.01),
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.pp_matrix.shape[1],),
                                    initializer=tf.keras.initializers.Zeros(),
                                    trainable=True)
        super(ReconstructedProteinLayer, self).build(input_shape)

    def call(self, inputs):
        protein_decode = tf.add(tf.matmul(inputs, tf.multiply(self.pp_matrix, self.kernel)),self.bias)
        protein_decode = tf.keras.activations.softplus(protein_decode)
        return protein_decode
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pp_matrix': self.pp_matrix.numpy().tolist()
        })
        return config


class ReconstructedGeneLayer(tf.keras.layers.Layer):

    """
        Custom layer for reconstructed gene decoding.

        This layer takes an input tensor and applies reconstructed gene decoding by performing matrix multiplication
        with the given gene-protein matrix and a learnable kernel.

        Parameters:
            gp_matrix (numpy.ndarray): Matrix for decoding gene-protein relationships.

        Returns:
            Tensor: Decoded gene representation.
    """

    def __init__(self, gp_matrix, **kwargs):
        super(ReconstructedGeneLayer, self).__init__(**kwargs)
        self.gp_matrix = tf.constant(gp_matrix, dtype=tf.float32)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1366, 1366),
                                      initializer=tf.keras.initializers.truncated_normal(stddev=0.01),
                                      trainable=True)

        super(ReconstructedGeneLayer, self).build(input_shape)

    def call(self, inputs):
        pheno_decode = tf.matmul(inputs, tf.multiply(self.gp_matrix, self.kernel))
        return pheno_decode
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'gp_matrix': self.gp_matrix.numpy().tolist()
        })
        return config


def build_model(gp_matrix, pp_matrix, t_gp_matrix, t_pp_matrix, input_shape):

    """
    Builds a model for protein-gene reconstruction using the given matrices.

    Parameters:
        gp_matrix (numpy.ndarray): Matrix for encoding protein-gene relationships.
        pp_matrix (numpy.ndarray): Matrix for encoding protein-phenotype relationships.
        t_gp_matrix (numpy.ndarray): Matrix for decoding protein-gene relationships.
        t_pp_matrix (numpy.ndarray): Matrix for decoding protein-phenotype relationships.
        input_shape (tuple): Shape of the input data.

    Returns:
        tf.keras.models.Model: The built model for protein-gene reconstruction.
    """


    # Define inputs

    input_layer = tf.keras.layers.Input(shape=input_shape)

    # Protein layer: Encodes the input gene expression data using the gene-protein mask
    protein_layer = ProteinLayer(gp_matrix=gp_matrix)(input_layer)

    # Phenotype layer: Encodes the protein layer output using the protein-phenotype mask
    pheno_layer = PhenoLayer(pp_matrix=pp_matrix)(protein_layer)

    # Reconstructed protein layer: Decodes the phenotype layer output using the transposed protein-phenotype mask
    reconstructed_protein_layer = ReconstructedProteinLayer(pp_matrix=t_pp_matrix)(pheno_layer)

    # Reconstructed gene layer: Decodes the reconstructed protein layer output using the transposed gene-protein mask
    reconstructed_gene_layer = ReconstructedGeneLayer(gp_matrix=t_gp_matrix)(reconstructed_protein_layer)

    # Define model
    model = tf.keras.models.Model(inputs=input_layer,
                                  outputs=reconstructed_gene_layer)

    return model
