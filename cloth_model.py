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

  def __init__(self, learned_model, name='Model', subequivariant=True):
    super(Model, self).__init__(name=name)

    self.subequivariant = subequivariant
    n_ouput_normalizer = 3

    if self.subequivariant:
      n_node_normalizer = 4+common.NodeType.SIZE
      n_edge_normalizer = 8 # 2D coord (2) + 3D coord (4) + 2*length = 8
    else:
      n_node_normalizer = 3+common.NodeType.SIZE # 12
      n_edge_normalizer = 7 # 2d coord (2) + 3D coord (3) + 2*length = 7

    with self._enter_variable_scope():
      self._learned_model = learned_model
      self._output_normalizer = normalization.Normalizer(
          size=n_ouput_normalizer, name='output_normalizer')
      self._node_normalizer = normalization.Normalizer(
          size=n_node_normalizer, name='node_normalizer')
      self._edge_normalizer = normalization.Normalizer(
          size=n_edge_normalizer, name='edge_normalizer')



  def _build_graph(self, inputs, is_training):
    """Builds input graph."""
    # construct graph nodes
    gravity_vector = tf.constant([0, 0, 1], dtype=tf.float32, shape=(1, 3, 1))

    velocity = inputs['world_pos'] - inputs['prev|world_pos'] # [Node, 3]
    node_count = tf.shape(velocity)[0]

    # construct graph edges
    senders, receivers = common.triangles_to_edges(inputs['cells'])
    relative_world_pos = (tf.gather(inputs['world_pos'], senders) -
                          tf.gather(inputs['world_pos'], receivers)) # [7000 (Edges) x 3 (Coords) x 2 ()]


    node_type = tf.one_hot(inputs['node_type'][:, 0], common.NodeType.SIZE)
    edge_count = tf.shape(relative_world_pos)[0]

    if self.subequivariant:
      # do the orgthogonalization
      velocity_g = tf.concat((tf.expand_dims(velocity, 2),
                              tf.repeat(gravity_vector, node_count, axis=0)), axis=-1) # shape [node, 3, 2]

      # Node x(M+1)x3 @ Node x3x(M+1) = Node x (M+1)x(M+1)
      velocity_orthog_inv = tf.einsum("nca,ncb->nab", velocity_g, velocity_g) # shape [Node, 2, 2]
      velocity_orthog_inv = tf.reshape(velocity_orthog_inv, [-1, 4]) # shape [Node, 4]

      node_features = tf.concat([velocity_orthog_inv, node_type], axis=-1) # shape [Node, 4+9=13]

      rel_world_pos_g = tf.concat((tf.expand_dims(relative_world_pos, 2), tf.repeat(gravity_vector, edge_count, axis=0)), axis=-1)
      rel_world_pos_orthog_inv = tf.einsum("eca,ecb->eab", rel_world_pos_g, rel_world_pos_g) # shape [7000 (edges), 2, 2]
      rel_world_pos = tf.reshape(rel_world_pos_orthog_inv, [-1, 4]) # shape [7000 (edges), 4]

    else:
      rel_world_pos = relative_world_pos
      node_features = tf.concat([velocity, node_type], axis=-1) # shape [Node, 3+9=12]


    relative_mesh_pos = (tf.gather(inputs['mesh_pos'], senders) -
                         tf.gather(inputs['mesh_pos'], receivers))
    edge_features = tf.concat([
        rel_world_pos,
        tf.norm(relative_world_pos, axis=-1, keepdims=True),
        relative_mesh_pos,
        tf.norm(relative_mesh_pos, axis=-1, keepdims=True)], axis=-1) # shape [7000 (edges), 8]

    mesh_edges = core_model.EdgeSet(
        name='mesh_edges',
        features=self._edge_normalizer(edge_features, is_training),
        receivers=receivers,
        senders=senders) # shape [7000 (edges), 8]


    return core_model.MultiGraph(
        node_features=self._node_normalizer(node_features, is_training), # shape [Node, 4+9=13]
        edge_sets=[mesh_edges])

  def _build(self, inputs):
    graph = self._build_graph(inputs, is_training=False)
    per_node_network_output = self._learned_model(graph) # [Node, 3]

    # transform back
    if self.subequivariant:
      network_output = tf.reshape(per_node_network_output, [-1, 3, 2])
      gravity_vector = tf.constant([0, 0, 1], dtype=tf.float32, shape=(1, 3, 1))
      velocity = inputs['world_pos'] - inputs['prev|world_pos'] # [Node, 3]
      node_count = tf.shape(velocity)[0]
      velocity_g = tf.concat((tf.expand_dims(velocity, 2), tf.repeat(gravity_vector, node_count, axis=0)), axis=-1) #[Node, 3]

      #eq. 9 somp:  [Z,g]@V_g(network output)
      transformed_output = tf.einsum("neg,ncg->nce", network_output, velocity_g) # [Node, 3 (coords), m' (3)]
      output = tf.math.reduce_mean(transformed_output, axis=-1) # [Node, 3 (coords)]
    else:
      output = per_node_network_output

    return self._update(inputs, output)

  @snt.reuse_variables
  def loss(self, inputs):
    """L2 loss on position."""
    graph = self._build_graph(inputs, is_training=True)
    network_output = self._learned_model(graph)

    # transform back
    if self.subequivariant:
      network_output = tf.reshape(network_output, [-1, 3, 2])
      gravity_vector = tf.constant([0, 0, 1], dtype=tf.float32, shape=(1, 3, 1))
      velocity = inputs['world_pos'] - inputs['prev|world_pos'] # [Node, 3]
      node_count = tf.shape(velocity)[0]
      velocity_g = tf.concat((tf.expand_dims(velocity, 2), tf.repeat(gravity_vector, node_count, axis=0)), axis=-1) #[Node, 3]

      #eq. 9 somp:  [Z,g]@V_g(network output)
      transformed_output = tf.einsum("neg,ncg->nce", network_output, velocity_g) # [Node, 3 (coords), m' (3)]
      output = tf.math.reduce_mean(transformed_output, axis=-1) # [Node, 3 (coords)]
    else:
      output = network_output

    # build target acceleration
    cur_position = inputs['world_pos']
    prev_position = inputs['prev|world_pos']
    target_position = inputs['target|world_pos']
    target_acceleration = target_position - 2*cur_position + prev_position
    target_normalized = self._output_normalizer(target_acceleration)

    # build loss
    loss_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.NORMAL)
    error = tf.reduce_sum((target_normalized - output)**2, axis=1)
    loss = tf.reduce_mean(error[loss_mask])
    return loss

  def _update(self, inputs, per_node_network_output):
    """Integrate model outputs."""
    acceleration = self._output_normalizer.inverse(per_node_network_output)
    # integrate forward
    cur_position = inputs['world_pos']
    prev_position = inputs['prev|world_pos']
    position = 2*cur_position + acceleration - prev_position
    return position
