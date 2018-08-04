# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf

from object_detection.meta_architectures import faster_rcnn_meta_arch
from nets import vgg

slim = tf.contrib.slim

class FasterRCNNVGG16FeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
  """Faster R-CNN with VGG16 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               batch_norm_trainable=False,
               reuse_weights=None,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16.
    """
    if first_stage_features_stride != 8 and first_stage_features_stride != 16:
      raise ValueError('`first_stage_features_stride` must be 8 or 16.')
    super(FasterRCNNVGG16FeatureExtractor, self).__init__(
        is_training, first_stage_features_stride, batch_norm_trainable,
        reuse_weights, weight_decay)

  def preprocess(self, resized_inputs):
    """Faster R-CNN with VGG16 preprocessing.

    Args:
      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.

    """
    return resized_inputs

  def _extract_proposal_features(self, preprocessed_inputs, scope):
    """Extracts first stage RPN features.

    Args:
      preprocessed_inputs: A [batch, height, width, channels] float32 tensor
        representing a batch of images.
      scope: A scope name.

    Returns:
      rpn_feature_map: A tensor with shape [batch, height, width, depth]
    Raises:
      InvalidArgumentError: If the spatial size of `preprocessed_inputs`
        (height or width) is less than 33.
      ValueError: If the created network is missing the required activation.
    """
    if len(preprocessed_inputs.get_shape().as_list()) != 4:
      raise ValueError('`preprocessed_inputs` must be 4 dimensional, got a '
                       'tensor of shape %s' % preprocessed_inputs.get_shape())

    with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=self._weight_decay)):
      with tf.variable_scope('vgg_16') as scope:
        net, endpoints = (
            vgg.vgg_16(
                preprocessed_inputs,
                num_classes=None, # Omit unneeded vgg logits layer
                is_training=self._is_training,
                scope=scope,
                fc_conv_padding='SAME'))
    return endpoints[scope.name + '/conv5/conv5_3']

  def _extract_box_classifier_features(self, proposal_feature_maps, scope):
    """Extracts second stage box classifier features.

    Args:
      proposal_feature_maps: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.
      scope: A scope name.

    Returns:
      proposal_classifier_features: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, height, width, depth]
        representing box classifier features for each proposal.
    """
    fc_conv_padding = 'SAME'
    dropout_keep_prob = 0.5
    is_training = self._is_training
    with tf.variable_scope('vgg_16'):
      with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=self._weight_decay)):
        # Copied from vgg.py onwards from conv_5
        net = slim.max_pool2d(proposal_feature_maps, [2, 2], scope='pool5')

        # Use conv2d instead of fully_connected layers.
        net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')

        """
        if global_pool:
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')

        if num_classes:
          net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                             scope='dropout7')
          net = slim.conv2d(net, num_classes, [1, 1],
                            activation_fn=None,
                            normalizer_fn=None,
                            scope='fc8')
        if spatial_squeeze and num_classes is not None:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        """

        return net

  def restore_from_classification_checkpoint_fn(
      self,
      first_stage_feature_extractor_scope,
      second_stage_feature_extractor_scope):
    """Returns a map of variables to load from a foreign checkpoint.

    Note that this overrides the default implementation in
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor.

    Args:
      first_stage_feature_extractor_scope: A scope name for the first stage
        feature extractor.
      second_stage_feature_extractor_scope: A scope name for the second stage
        feature extractor.

    Returns:
      A dict mapping variable names (to load from a checkpoint) to variables in
      the model graph.
    """
    variables_to_restore = {}
    for variable in tf.global_variables():
      if variable.op.name.startswith(
          first_stage_feature_extractor_scope):
        # imagenet
        #var_name = variable.op.name.replace(first_stage_feature_extractor_scope + '/', '')

        # tmv (should also do fc1 -> fc6, but is different shape...)
        s = variable.op.name.split('/')
        var_name = '/'.join(s[-2:])

        variables_to_restore[var_name] = variable
      if variable.op.name.startswith(
          second_stage_feature_extractor_scope):
        # imagenet
        #var_name = variable.op.name.replace(second_stage_feature_extractor_scope + '/', '')

        # tmv (should also do fc1 -> fc6, but is different shape...)
        s = variable.op.name.split('/')
        var_name = '/'.join(s[-2:])

        variables_to_restore[var_name] = variable
    return variables_to_restore

