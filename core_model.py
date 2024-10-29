# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Core learned graph net model."""

import collections
import functools

import sonnet as snt
import tensorflow.compat.v1 as tf

EdgeSet = collections.namedtuple(
    "EdgeSet", ["name", "features", "senders", "receivers"]
)
MultiGraph = collections.namedtuple("Graph", ["node_features", "edge_sets"])


class GraphNetBlock(snt.AbstractModule):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, model_fn, name="GraphNetBlock"):
        super().__init__(name=name)
        self._model_fn = model_fn

    def _update_edge_features(self, node_features, edge_set):
        """Aggregrates node features, and applies edge function."""
        sender_features = tf.gather(node_features, edge_set.senders)
        receiver_features = tf.gather(node_features, edge_set.receivers)
        features = [sender_features, receiver_features, edge_set.features]
        with tf.variable_scope(edge_set.name + "_edge_fn"):
            return self._model_fn(
                neighbours=len(features),
                objects=9212,  # TODO: hard coded object count
            )(tf.concat(features, axis=-1))

    def _update_node_features(self, node_features, edge_sets):
        """Aggregrates edge features, and applies node function."""
        num_nodes = tf.shape(node_features)[0]
        features = [node_features]
        for edge_set in edge_sets:
            features.append(
                tf.math.unsorted_segment_sum(
                    edge_set.features, edge_set.receivers, num_nodes
                )
            )
        with tf.variable_scope("node_fn"):
            return self._model_fn(
                neighbours=len(features),
                objects=1579,  # TODO: hard coded object count
            )(tf.concat(features, axis=-1))

    def _build(self, graph):
        """Applies GraphNetBlock and returns updated MultiGraph."""

        # apply edge functions
        new_edge_sets = []
        for edge_set in graph.edge_sets:
            updated_features = self._update_edge_features(graph.node_features, edge_set)
            new_edge_sets.append(edge_set._replace(features=updated_features))

        # apply node function
        new_node_features = self._update_node_features(
            graph.node_features, new_edge_sets
        )

        # add residual connections
        new_node_features += graph.node_features
        new_edge_sets = [
            es._replace(features=es.features + old_es.features)
            for es, old_es in zip(new_edge_sets, graph.edge_sets)
        ]
        return MultiGraph(new_node_features, new_edge_sets)


class InvarianceTransform(snt.AbstractModule):
    """Wrap a network with an invariance transform around certain parts of the latent"""

    def __init__(
        self,
        network,
        in_z_size: int,
        in_h_size: int,
        out_z_size: int,
        out_h_size: int,
        neighbours: int,
        objects: int,
        name="InvarianceTransform",
    ):
        super().__init__(name=name)
        self.network = network
        self.neighbours = neighbours
        self.objects = objects
        assert in_z_size % 3 == 0, "in_z_size must be a multiple of 3"
        self._in_z_size = in_z_size
        self._in_h_size = in_h_size

        assert out_z_size % 3 == 0, "out_z_size must be a multiple of 3"
        self._out_z_size = out_z_size
        self._out_h_size = out_h_size

    def _build(self, latent):
        # In shape: [Batch * Latent (in_Z + in_h)]
        # Out shape: [New latent (out_Z + out_h)]
        latent.set_shape((self.objects, self.neighbours * 64))
        latent = tf.reshape(
            latent, (self.objects, self.neighbours, self._in_z_size + self._in_h_size)
        )
        gravity_vector = tf.constant([0, 0, 1], dtype=tf.float32, shape=(1, 1, 3))
        m = self._in_z_size // 3  # Number of 3D vectors in Z == `m` from SOMP
        m_prime = self._out_z_size // 3  # Number of 3D vectors in output
        in_z = tf.reshape(
            latent[:, :, : self._in_z_size], (self.objects, m * self.neighbours, 3)
        )  # [Node, m * neighbours, Coordinates]
        in_h = tf.reshape(
            latent[:, :, self._in_z_size :],
            (self.objects, self._in_h_size * self.neighbours),
        )  # [Node, h * neighbours]

        z_g = tf.concat(
            (tf.repeat(gravity_vector, latent.shape[0], axis=0), in_z), axis=1
        )  # [nodes, 1+(m*neighbours), 3]

        z_orthogonal = tf.einsum(
            "nac,nbc->nab", z_g, z_g
        )  # [Node, (m*neighbours)+1, (m*neighbours)+1]

        z_orthogonal_flat = tf.reshape(
            z_orthogonal, (self.objects, (m * self.neighbours + 1) ** 2)
        )

        net_in = tf.reshape(
            tf.concat([z_orthogonal_flat, in_h], axis=1),
            (
                self.objects,
                (m * self.neighbours + 1) ** 2 + self._in_h_size * self.neighbours,
            ),
        )

        net_out = self.network(net_in)  # Network output, called `V_g` in SOMP
        assert net_out.shape == tf.TensorShape(
            (
                self.objects,
                (m + 1) * m_prime + self._out_h_size,
            )
        ), f"Strange V_g shape {net_out.shape}"

        out_z = tf.reshape(
            net_out[:, : -self._out_h_size],
            (self.objects, (m + 1), m_prime),
        )

        out_h = net_out[:, -self._out_h_size :]

        # The first object must correspond with 'ourselves', so we take gravity + this
        out_z_transformed = tf.einsum("nmc,nmb->nbc", z_g[:, : m + 1], out_z)
        assert out_z_transformed.shape == tf.TensorShape(
            (self.objects, m_prime, 3)
        ), out_z_transformed.shape

        out_z_flat = tf.reshape(
            out_z_transformed,
            (self.objects, self._out_z_size),
        )

        output = tf.concat((out_z_flat, out_h), axis=1)
        assert output.shape == tf.TensorShape((self.objects, 64)), output.shape

        return output


class EncodeProcessDecode(snt.AbstractModule):
    """Encode-Process-Decode GraphNet model."""

    def __init__(
        self,
        output_size,
        latent_size,
        num_layers,
        message_passing_steps,
        name="EncodeProcessDecode",
        subeq_layers=False,
    ):
        super().__init__(name=name)
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._message_passing_steps = message_passing_steps
        self._subeq_layers = subeq_layers

    def _make_mlp(
        self,
        output_size: int,
        layer_norm: bool = True,
        subequivariant: bool = False,
        neighbours: int = None,
        objects: int = None,
    ):
        """Builds an MLP."""
        if subequivariant:
            assert (
                output_size == 64
            ), "This code assumes a output of 64 for subequivariant layers"
            m = 16  # `m` for input and `m_prime` for output are equal here
            h_size = output_size - m * 3
            widths = [self._latent_size] * self._num_layers + [(m + 1) * m + h_size]
            network = snt.nets.MLP(widths, activate_final=False)
            network = InvarianceTransform(
                network,
                in_z_size=m * 3,
                in_h_size=h_size,
                out_z_size=m * 3,
                out_h_size=h_size,
                neighbours=neighbours,
                objects=objects,
            )
            assert m * 3 + h_size == 64
        else:
            widths = [self._latent_size] * self._num_layers + [output_size]
            network = snt.nets.MLP(widths, activate_final=False)

        if layer_norm:
            network = snt.Sequential([network, snt.LayerNorm()])
        return network

    def _encoder(self, graph):
        """Encodes node and edge features into latent features."""
        with tf.variable_scope("encoder"):
            node_latents = self._make_mlp(self._latent_size)(graph.node_features)
            new_edges_sets = []
            for edge_set in graph.edge_sets:
                latent = self._make_mlp(self._latent_size)(edge_set.features)
                new_edges_sets.append(edge_set._replace(features=latent))
        return MultiGraph(node_latents, new_edges_sets)

    def _decoder(self, graph):
        """Decodes node features from graph."""
        with tf.variable_scope("decoder"):
            decoder = self._make_mlp(self._output_size, layer_norm=False)
            return decoder(graph.node_features)

    def _build(self, graph):
        """Encodes and processes a multigraph, and returns node features."""
        model_fn = functools.partial(
            self._make_mlp,
            output_size=self._latent_size,
            subequivariant=self._subeq_layers,
        )
        latent_graph = self._encoder(graph)
        for _ in range(self._message_passing_steps):
            latent_graph = GraphNetBlock(model_fn)(latent_graph)
        return self._decoder(latent_graph)
