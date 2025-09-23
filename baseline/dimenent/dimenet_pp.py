import tensorflow as tf

from .layers.embedding_block import EmbeddingBlock
from .layers.bessel_basis_layer import BesselBasisLayer
from .layers.spherical_basis_layer import SphericalBasisLayer
from .layers.interaction_pp_block import InteractionPPBlock
from .layers.output_pp_block import OutputPPBlock
from tensorflow.keras import layers
from .activations import swish


class DimeNetPP_original(tf.keras.Model):
   
    def __init__(
            self, emb_size, out_emb_size, int_emb_size, basis_emb_size,
            num_blocks, num_spherical, num_radial,
            cutoff=5.0, envelope_exponent=5, num_before_skip=1,
            num_after_skip=2, num_dense_output=3, num_labels=12,
            activation=swish, extensive=True, output_init='zeros',
            name='dimenet', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_blocks = num_blocks
        self.extensive = extensive

        # Radial and spherical basis layers
        self.rbf_layer = BesselBasisLayer(
            num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)
        self.sbf_layer = SphericalBasisLayer(
            num_spherical, num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)

        # Embedding and output blocks
        self.output_blocks = []
        self.emb_block = EmbeddingBlock(emb_size, activation=activation)
        self.output_blocks.append(
            OutputPPBlock(emb_size, out_emb_size, num_dense_output, num_labels,
                          activation=activation, output_init=output_init))

        # Interaction blocks
        self.int_blocks = []
        for i in range(num_blocks):
            self.int_blocks.append(
                InteractionPPBlock(emb_size, int_emb_size, basis_emb_size, num_before_skip,
                                   num_after_skip, activation=activation))
            self.output_blocks.append(
                OutputPPBlock(emb_size, out_emb_size, num_dense_output, num_labels,
                              activation=activation, output_init=output_init))

    def calculate_interatomic_distances(self, R, idx_i, idx_j):
        """Compute pairwise interatomic distances."""
        Ri = tf.gather(R, idx_i)
        Rj = tf.gather(R, idx_j)
        Dij = tf.sqrt(tf.nn.relu(tf.reduce_sum((Ri - Rj)**2, -1)))
        return Dij

    def calculate_neighbor_angles(self, R, id3_i, id3_j, id3_k):
        """Compute angles between neighboring atoms (triplets)."""
        Ri = tf.gather(R, id3_i)
        Rj = tf.gather(R, id3_j)
        Rk = tf.gather(R, id3_k)
        R1 = Rj - Ri
        R2 = Rk - Rj
        x = tf.reduce_sum(R1 * R2, axis=-1)
        y = tf.linalg.cross(R1, R2)
        y = tf.norm(y, axis=-1)
        angle = tf.math.atan2(y, x)
        return angle

    def call(self, inputs):
        # Extract inputs
        Z, R = inputs['Z'], inputs['R']
        batch_seg = inputs['batch_seg']
        idnb_i, idnb_j = inputs['idnb_i'], inputs['idnb_j']
        id_expand_kj, id_reduce_ji = inputs['id_expand_kj'], inputs['id_reduce_ji']
        id3dnb_i, id3dnb_j, id3dnb_k = inputs['id3dnb_i'], inputs['id3dnb_j'], inputs['id3dnb_k']
        n_atoms = tf.shape(Z)[0]

        # Compute distances and basis functions
        Dij = self.calculate_interatomic_distances(R, idnb_i, idnb_j)
        rbf = self.rbf_layer(Dij)
        Anglesijk = self.calculate_neighbor_angles(R, id3dnb_i, id3dnb_j, id3dnb_k)
        sbf = self.sbf_layer([Dij, Anglesijk, id_expand_kj])

        # Embedding and initial output
        x = self.emb_block([Z, rbf, idnb_i, idnb_j])
        P = self.output_blocks[0]([x, rbf, idnb_i, n_atoms])

        # Interaction blocks and subsequent outputs
        for i in range(self.num_blocks):
            x = self.int_blocks[i]([x, rbf, sbf, id_expand_kj, id_reduce_ji])
            P += self.output_blocks[i+1]([x, rbf, idnb_i, n_atoms])

        # Aggregate outputs by batch
        if self.extensive:
            P = tf.math.segment_sum(P, batch_seg)
        else:
            P = tf.math.segment_mean(P, batch_seg)
        return P 


class DimeNetPP(DimeNetPP_original):
    def __init__(
            self, emb_size=128, out_emb_size=256, int_emb_size=64, basis_emb_size=8,
            num_blocks=4, num_spherical=7, num_radial=6,
            cutoff=5.0, envelope_exponent=5, num_before_skip=1,
            num_after_skip=2, num_dense_output=3, num_labels=48,
            activation=swish, extensive=False, output_init='glorot_normal',
            name='dimenet', **kwargs):
        
        super().__init__(
            emb_size=emb_size, out_emb_size=out_emb_size, int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size, num_blocks=num_blocks,
            num_spherical=num_spherical, num_radial=num_radial,
            cutoff=cutoff, envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip, num_after_skip=num_after_skip,
            num_dense_output=num_dense_output,
            activation=activation, extensive=extensive, output_init=output_init,
            name=name, **kwargs
        )

        self.num_labels = num_labels
        self.extensive = extensive
        output_init = tf.keras.initializers.Orthogonal(seed=42)

        # Define classification heads for 48 binary labels, outputting softmax probabilities
        self.classifiers = [
            layers.Dense(2, activation='softmax', kernel_initializer=output_init,
                         name=f'class_head_{i}') for i in range(num_labels)
        ]

    def call(self, inputs):
        # Obtain molecule-level and atom-level embeddings
        P = super().call(inputs)  # P: [batch, out_emb_size], x: atom-level embedding
        # Apply multiple classification heads
        outputs = [clf(P) for clf in self.classifiers]  # each [batch, 2]
        return tf.stack(outputs, axis=1)  # Returns [batch, num_labels, 2] and embedding x

