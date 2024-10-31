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
"""Model for FlagSimple."""

import sonnet as snt
import tensorflow.compat.v1 as tf

import common
import core_model
import normalization


class Model(snt.AbstractModule):
    """Model for static cloth simulation."""

    def __init__(self, learned_model, name="Model", subeq_model=True):
        super().__init__(name=name)

        self.subeq_model = subeq_model
        n_ouput_normalizer = 3

        if self.subeq_model:
            n_node_normalizer = 4 + common.NodeType.SIZE
            n_edge_normalizer = 8  # 2D coord (2) + 3D coord (4) + 2*length = 8
        else:
            n_node_normalizer = 3 + common.NodeType.SIZE  # 12
            n_edge_normalizer = 7  # 2d coord (2) + 3D coord (3) + 2*length = 7

        with self._enter_variable_scope():
            self._learned_model = learned_model
            self._output_normalizer = normalization.Normalizer(
                size=n_ouput_normalizer, name="output_normalizer"
            )
            self._node_normalizer = normalization.Normalizer(
                size=n_node_normalizer, name="node_normalizer"
            )
            self._edge_normalizer = normalization.Normalizer(
                size=n_edge_normalizer, name="edge_normalizer"
            )

    @staticmethod
    def _subequivariant_transform(coordinate_array):
        # Input shape: [? (object), 3 (coords)]
        # Output shape: [? (object), 4 (2 * (m+1))]
        gravity_vector = tf.constant([0, 0, 1], dtype=tf.float32, shape=(1, 3, 1))
        object_count = tf.shape(coordinate_array)[0]
        coord_g = tf.concat(
            (
                tf.expand_dims(coordinate_array, 2),
                tf.repeat(gravity_vector, object_count, axis=0),
            ),
            axis=-1,
        )  # shape [?, 3, 2]
        # [? x(M+1)x3] @ [? x3x(M+1)] = [? x (M+1)x(M+1)]
        coord_orthog = tf.einsum("nca,ncb->nab", coord_g, coord_g)
        return tf.reshape(coord_orthog, [object_count, -1])

    def _build_graph(self, inputs, is_training):
        """Builds input graph."""
        # construct graph nodes
        node_velocity = inputs["world_pos"] - inputs["prev|world_pos"]  # [Node, 3]
        node_type = tf.one_hot(inputs["node_type"][:, 0], common.NodeType.SIZE)

        # construct graph edges
        senders, receivers = common.triangles_to_edges(inputs["cells"])
        edge_rel_world_pos = tf.gather(inputs["world_pos"], senders) - tf.gather(
            inputs["world_pos"], receivers
        )  # [9000 (Edges) x 3 (Coords)]
        edge_relative_mesh_pos = tf.gather(inputs["mesh_pos"], senders) - tf.gather(
            inputs["mesh_pos"], receivers
        )

        if self.subeq_model:
            edge_rel_world_pos = self._subequivariant_transform(edge_rel_world_pos)
            node_velocity = self._subequivariant_transform(node_velocity)

        node_features = tf.concat(
            [node_velocity, node_type], axis=-1
        )  # shape [1000 (nodes), (3 or 4)+9=(12 or 13)]

        edge_features = tf.concat(
            [
                edge_rel_world_pos,
                tf.norm(edge_rel_world_pos, axis=-1, keepdims=True),
                edge_relative_mesh_pos,
                tf.norm(edge_relative_mesh_pos, axis=-1, keepdims=True),
            ],
            axis=-1,
        )  # shape [7000 (edges), 8]

        edge_set = core_model.EdgeSet(
            name="mesh_edges",
            features=self._edge_normalizer(edge_features, is_training),
            receivers=receivers,
            senders=senders,
        )

        return core_model.MultiGraph(
            node_features=self._node_normalizer(node_features, is_training),
            edge_sets=[edge_set],
        )

    @staticmethod
    def _subequivariant_transform_back(inputs, per_node_network_output):
        network_output = tf.reshape(
            per_node_network_output, [-1, 3, 2]
        )  # [Node, 3 (m'), 2 (m+1)]
        gravity_vector = tf.constant(
            [0, 0, 1], dtype=tf.float32, shape=(1, 3, 1)
        )  # [1 (will be repeated to Node), 3 (coords), 1 (the +1 from m+1)]
        velocity = inputs["world_pos"] - inputs["prev|world_pos"]  # [Node, 3]
        node_count = tf.shape(velocity)[0]  # (1,)
        velocity_g = tf.concat(
            (
                tf.expand_dims(velocity, 2),
                tf.repeat(gravity_vector, node_count, axis=0),
            ),
            axis=-1,
        )  # [Node, 3 (coords), 2 (m+1)]

        # eq. 9 somp:  [Z,g]@V_g(network output)
        transformed_output = tf.einsum(
            "nmg,ncg->ncm", network_output, velocity_g
        )  # [Node, 3 (coords), m' (3)]

        return tf.math.reduce_mean(transformed_output, axis=2)  # [Node, 3 (coords)]

    def _build(self, inputs):
        graph = self._build_graph(inputs, is_training=False)
        network_output = self._learned_model(graph)  # [Node, 3]

        # transform back
        if self.subeq_model:
            network_output = self._subequivariant_transform_back(inputs, network_output)

        return self._update(inputs, network_output)

    @snt.reuse_variables
    def loss(self, inputs):
        """L2 loss on position."""
        graph = self._build_graph(inputs, is_training=True)
        network_output = self._learned_model(graph)

        # transform back
        if self.subeq_model:
            network_output = self._subequivariant_transform_back(inputs, network_output)

        # build target acceleration
        cur_position = inputs["world_pos"]
        prev_position = inputs["prev|world_pos"]
        target_position = inputs["target|world_pos"]
        target_acceleration = target_position - 2 * cur_position + prev_position
        target_normalized = self._output_normalizer(target_acceleration)

        # build loss
        loss_mask = tf.equal(inputs["node_type"][:, 0], common.NodeType.NORMAL)
        error = tf.reduce_sum((target_normalized - network_output) ** 2, axis=1)
        loss = tf.reduce_mean(error[loss_mask])
        return loss

    def _update(self, inputs, per_node_network_output):
        """Integrate model outputs."""
        acceleration = self._output_normalizer.inverse(per_node_network_output)
        # integrate forward
        cur_position = inputs["world_pos"]
        prev_position = inputs["prev|world_pos"]
        position = 2 * cur_position + acceleration - prev_position
        return position
