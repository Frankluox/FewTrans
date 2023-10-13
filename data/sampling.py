"""
Data sampling for both episodic and non-episodic training/testing.
Adapted from original Meta-Dataset code.
"""

# coding=utf-8
# Copyright 2022 The Meta-Dataset Authors.
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

from .dataset_spec import get_classes,create_spec, Split
import numpy as np
from .ImageNet_graph_operations import get_leaves, get_spanning_leaves
import torch
import logging
import collections
from PIL import Image
MAX_SPANNING_LEAVES_ELIGIBLE = 392




  
  
    



def sample_num_ways_uniformly(num_classes, min_ways, max_ways, rng=None):
    """Samples a number of ways for an episode uniformly and at random.
    The support of the distribution is [min_ways, num_classes], or
    [min_ways, max_ways] if num_classes > max_ways.
    Args:
    num_classes: int, number of classes.
    min_ways: int, minimum number of ways.
    max_ways: int, maximum number of ways. Only used if num_classes > max_ways.
    rng: np.random.RandomState used for sampling.
    Returns:
    num_ways: int, number of ways for the episode.
    """
    rng = rng or RNG
    max_ways = min(max_ways, num_classes)
    sample_ways = rng.randint(low=min_ways, high=max_ways + 1)
    return sample_ways


def sample_class_ids_uniformly(num_ways, rel_classes, rng=None):
    """Samples the (relative) class IDs for the episode.
    Args:
    num_ways: int, number of ways for the episode.
    rel_classes: list of int, available class IDs to sample from.
    rng: np.random.RandomState used for sampling.
    Returns:
    class_ids: np.array, class IDs for the episode, with values in rel_classes.
    """
    rng = rng or RNG
    return rng.choice(rel_classes, num_ways, replace=False)


def compute_num_query(images_per_class, max_num_query, num_support):
  """Computes the number of query examples per class in the episode.
  Query sets are balanced, i.e., contain the same number of examples for each
  class in the episode.
  The number of query examples satisfies the following conditions:
  - it is no greater than `max_num_query`
  - if support size is unspecified, it is at most half the size of the
    smallest class in the episode
  - if support size is specified, it is at most the size of the smallest class
    in the episode minus the max support size.
  Args:
    images_per_class: np.array, number of images for each class.
    max_num_query: int, number of images for each class.
    num_support: int or tuple(int, int), number (or range) of support
      images per class.
  Returns:
    num_query: int, number of query examples per class in the episode.
  """
  if num_support is None:
    if images_per_class.min() < 2:
      raise ValueError('Expected at least 2 images per class.')
    return np.minimum(max_num_query, (images_per_class // 2).min())
  elif isinstance(num_support, int):
    max_support = num_support
  else:
    _, max_support = num_support
  if (images_per_class - max_support).min() < 1:
    raise ValueError(
        'Expected at least {} images per class'.format(max_support + 1))
  return np.minimum(max_num_query, images_per_class.min() - max_support)


def sample_support_set_size(num_remaining_per_class,
                            max_support_size_contrib_per_class,
                            max_support_set_size,
                            rng=None):
  """Samples the size of the support set in the episode.
  That number is such that:
  * The contribution of each class to the number is no greater than
    `max_support_size_contrib_per_class`.
  * It is no greater than `max_support_set_size`.
  * The support set size is greater than or equal to the number of ways.
  Args:
    num_remaining_per_class: np.array, number of images available for each class
      after taking into account the number of query images.
    max_support_size_contrib_per_class: int, maximum contribution for any given
      class to the support set size. Note that this is not a limit on the number
      of examples of that class in the support set; this is a limit on its
      contribution to computing the support set _size_.
    max_support_set_size: int, maximum size of the support set.
    rng: np.random.RandomState used for sampling.
  Returns:
    support_set_size: int, size of the support set in the episode.
  """
  rng = rng or RNG
  if max_support_set_size < len(num_remaining_per_class):
    raise ValueError('max_support_set_size is too small to have at least one '
                     'support example per class.')
  beta = rng.uniform()
  # print('hh')
  # print(beta)
  

  support_size_contributions = np.minimum(max_support_size_contrib_per_class,
                                          num_remaining_per_class)
  # print(support_size_contributions)
  # print(np.floor(beta * support_size_contributions + 1).sum())
  return np.minimum(
      # Taking the floor and adding one is equivalent to sampling beta uniformly
      # in the (0, 1] interval and taking the ceiling of its product with
      # `support_size_contributions`. This ensures that the support set size is
      # at least as big as the number of ways.
      np.floor(beta * support_size_contributions + 1).sum(),
      max_support_set_size)


def sample_num_support_per_class(images_per_class,
                                 num_remaining_per_class,
                                 support_set_size,
                                 max_support_size_contrib_per_class,
                                 min_log_weight,
                                 max_log_weight,
                                 rng=None):
  """Samples the number of support examples per class.
  At a high level, we wish the composition to loosely match class frequencies.
  Sampling is done such that:
  * The number of support examples per class is no greater than
    `support_set_size`.
  * The number of support examples per class is no greater than the number of
    remaining examples per class after the query set has been taken into
    account.
  Args:
    images_per_class: np.array, number of images for each class.
    num_remaining_per_class: np.array, number of images available for each class
      after taking into account the number of query images.
    support_set_size: int, size of the support set in the episode.
    min_log_weight: float, minimum log-weight to give to any particular class.
    max_log_weight: float, maximum log-weight to give to any particular class.
    rng: np.random.RandomState used for sampling.
  Returns:
    num_support_per_class: np.array, number of support examples for each class.
  """
  rng = rng or RNG
  if support_set_size < len(num_remaining_per_class):
    raise ValueError('Requesting smaller support set than the number of ways.')
  if np.min(num_remaining_per_class) < 1:
    raise ValueError('Some classes have no remaining examples.')

  # Remaining number of support examples to sample after we guarantee one
  # support example per class.
  remaining_support_set_size = support_set_size - len(num_remaining_per_class)

  unnormalized_proportions = images_per_class * np.exp(
      rng.uniform(min_log_weight, max_log_weight, size=images_per_class.shape))
  support_set_proportions = (
      unnormalized_proportions / unnormalized_proportions.sum())

  # This guarantees that there is at least one support example per class.
  num_desired_per_class = np.floor(
      support_set_proportions * remaining_support_set_size).astype('int32') + 1
  num_desired_per_class = np.minimum(num_desired_per_class, max_support_size_contrib_per_class)

  return np.minimum(num_desired_per_class, num_remaining_per_class)




class EpisodeSampler(object):
  """Generates samples of an episode.
  In particular, for each episode, it will sample all files and labels of a task.
  """

  def __init__(self,
               seed,
               dataset_spec,
               episode_descr_config,
               base2novel = False,
               eval_only = False
               ):
    """
    seed: seed for sampling
    dataset_spec: dataset specification
    split: which split to sample from
    episode_descr_config: detailed configurations about how to sample a episode
    """

    # Fixing seed for sampling
    self._rng = np.random.RandomState(seed)


    self.dataset_spec = dataset_spec
    self.base2novel = base2novel
    self.eval_only = eval_only

    self.num_ways = episode_descr_config.NUM_WAYS
    self.num_support = episode_descr_config.NUM_SUPPORT
    self.num_query_base = episode_descr_config.NUM_QUERY_BASE
    self.num_query_novel = episode_descr_config.NUM_QUERY_NOVEL
    self.min_ways = episode_descr_config.MIN_WAYS
    self.max_ways_upper_bound = episode_descr_config.MAX_WAYS_UPPER_BOUND
    self.max_num_query = episode_descr_config.MAX_NUM_QUERY
    self.max_support_set_size = episode_descr_config.MAX_SUPPORT_SET_SIZE
    self.max_support_size_contrib_per_class = episode_descr_config.MAX_SUPPORT_SIZE_CONTRIB_PER_CLASS
    self.min_log_weight = episode_descr_config.MIN_LOG_WEIGHT
    self.max_log_weight = episode_descr_config.MAX_LOG_WEIGHT
    self.min_examples_in_class = episode_descr_config.MIN_EXAMPLES_IN_CLASS
    self.use_dag_hierarchy = episode_descr_config.USE_DAG_HIERARCHY
    self.sample_all = episode_descr_config.SAMPLE_ALL

    self.base_class_set = get_classes(Split.BASE, self.dataset_spec["num_classes_per_split"])
    self.novel_class_set = get_classes(Split.NOVEL, self.dataset_spec["num_classes_per_split"])


    self.num_base_classes = len(self.base_class_set)
    self.num_novel_classes = len(self.novel_class_set)
    # Filter out classes with too few examples
    self._filtered_base_class_set = []
    # Store (class_id, n_examples) of skipped classes for logging.
    skipped_classes = []
    
    for class_id in self.base_class_set:
      n_examples = len(dataset_spec["images_per_class"][class_id])

      if n_examples < self.min_examples_in_class:
        skipped_classes.append((class_id, n_examples))
      else:
        self._filtered_base_class_set.append(class_id)
    self.num_filtered_base_classes = len(self._filtered_base_class_set)

    if skipped_classes:
      logging.info(
          'Skipping the following classes, which do not have at least '
          '%d examples', self.min_examples_in_class)
    for class_id, n_examples in skipped_classes:
      logging.info('%s (ID=%d, %d examples)',
                   dataset_spec["id2name"][class_id], class_id, n_examples)

    if self.min_ways and self.num_filtered_base_classes < self.min_ways:
      raise ValueError(
          '"min_ways" is set to {}, but base set of dataset {} only has {} '
          'classes with at least {} examples ({} total), so it is not possible '
          'to create an episode for it. This may have resulted from applying a '
          'restriction on this split of this dataset by specifying '
          'benchmark.restrict_classes or benchmark.min_examples_in_class.'
          .format(self.min_ways, dataset_spec["name"],
                  self.num_filtered_base_classes, self.min_examples_in_class,
                  self.num_base_classes))


    # for ImageNet
    if self.dataset_spec["name"] == "ILSVRC" and self.use_dag_hierarchy:
      if self.num_ways is not None:
        raise ValueError('"use_dag_hierarchy" is incompatible with "num_ways".')

      if not dataset_spec["name"] == "ILSVRC":
        raise ValueError('Only applicable to ImageNet.')

      # A DAG for navigating the ontology for the given split.
      base_graph = dataset_spec["split_subgraph"][Split.BASE]
      novel_graph = dataset_spec["split_subgraph"][Split.NOVEL]

      # Map the absolute class IDs in the split's class set to IDs relative to
      # the split.

      abs_to_rel_ids_base = dict((abs_id, i) for i, abs_id in enumerate(self.base_class_set))
      abs_to_rel_ids_novel = dict((abs_id, i) for i, abs_id in enumerate(self.novel_class_set))

      # Extract the sets of leaves and internal nodes in the DAG.
      base_leaves = set(get_leaves(base_graph))
      novel_leaves = set(get_leaves(novel_graph))
      base_internal_nodes = base_graph - base_leaves  # set difference
      novel_internal_nodes = novel_graph - novel_leaves  # set difference

      # Map each node of the DAG to the Synsets of the leaves it spans.
      base_spanning_leaves_dict = get_spanning_leaves(base_graph)
      novel_spanning_leaves_dict = get_spanning_leaves(novel_graph)

      # Build a list of lists storing the relative class IDs of the spanning
      # leaves for each eligible internal node. We ensure a deterministic order
      # by sorting the inner-nodes and their corresponding leaves by wn_id.
      self.span_leaves_rel = collections.defaultdict(list)
      
      for node in sorted(base_internal_nodes, key=lambda n: n.wn_id):
        node_leaves = sorted(base_spanning_leaves_dict[node], key=lambda n: n.wn_id)
        # Build a list of relative class IDs of leaves that have at least
        # min_examples_in_class examples.
        ids_rel_base = []
        for leaf in node_leaves:
          abs_id = dataset_spec["class_names_to_ids"][leaf.wn_id]
          if abs_id in self._filtered_base_class_set:
            ids_rel_base.append(abs_to_rel_ids_base[abs_id])

        # Internal nodes are eligible if they span at least
        # `min_allowed_classes` and at most `max_eligible` leaves.
        if self.min_ways <= len(ids_rel_base) <= MAX_SPANNING_LEAVES_ELIGIBLE:
          self.span_leaves_rel["base"].append(ids_rel_base)

      for node in sorted(novel_internal_nodes, key=lambda n: n.wn_id):
        node_leaves = sorted(novel_spanning_leaves_dict[node], key=lambda n: n.wn_id)
        # Build a list of relative class IDs of leaves that have at least
        # min_examples_in_class examples.

        ids_rel_novel = []
        for leaf in node_leaves:
          abs_id = dataset_spec["class_names_to_ids"][leaf.wn_id]

          if abs_id in self.novel_class_set:
            ids_rel_novel.append(abs_to_rel_ids_novel[abs_id])

        # Internal nodes are eligible if they span at least
        # `min_allowed_classes` and at most `max_eligible` leaves.
        if self.min_ways <= len(ids_rel_novel) <= MAX_SPANNING_LEAVES_ELIGIBLE:
          self.span_leaves_rel["novel"].append(ids_rel_novel)

      num_eligible_nodes_base = len(self.span_leaves_rel["base"])
      num_eligible_nodes_novel = len(self.span_leaves_rel["novel"])
      if num_eligible_nodes_base < 1 or num_eligible_nodes_novel < 1:
        raise ValueError('There are no classes eligible for participating in '
                         'episodes. Consider changing the value of '
                         '`EpisodeDescriptionSampler.min_ways`, or '
                         'or MAX_SPANNING_LEAVES_ELIGIBLE')

                

  def sample_class_ids(self, sample_set):
    """Returns the (relative) class IDs for an episode.
    If self.min_examples_in_class > 0, classes with too few examples will not
    be selected.
    """
    if self.dataset_spec["name"] == "ILSVRC" and self.use_dag_hierarchy:
      assert not self.sample_all
      # Retrieve the list of relative class IDs for an internal node sampled
      # uniformly at random.
      # span_leaves_rel = self.span_leaves_rel[sample_set]
      index = self._rng.choice(list(range(len(self.span_leaves_rel[sample_set]))))
      episode_classes_rel = self.span_leaves_rel[sample_set][index]

      # If the number of chosen classes is larger than desired, sub-sample them.
      if len(episode_classes_rel) > self.max_ways_upper_bound:
        episode_classes_rel = self._rng.choice(
            episode_classes_rel,
            size=[self.max_ways_upper_bound],
            replace=False)

      # Light check to make sure the chosen number of classes is valid.
      assert len(episode_classes_rel) >= self.min_ways
      assert len(episode_classes_rel) <= self.max_ways_upper_bound
    else:
      if self.num_ways is not None:
          num_ways = self.num_ways
      else:
          num_classes = self.num_filtered_base_classes if sample_set=="base" else self.num_novel_classes
          num_ways = sample_num_ways_uniformly(
              num_classes,
              min_ways=self.min_ways,
              max_ways=self.max_ways_upper_bound,
              rng=self._rng)
      # Filtered class IDs relative to the selected split
      start_ID = self.base_class_set[0] if sample_set=="base" else self.novel_class_set[0]
      class_set = self._filtered_base_class_set if sample_set=="base" else self.novel_class_set
      ids_rel = [
          class_id - start_ID for class_id in class_set
      ]
      if self.sample_all:
        return ids_rel
      episode_classes_rel = sample_class_ids_uniformly(
          num_ways, ids_rel, rng=self._rng)

    return episode_classes_rel


  def sample_single_episode(self):
      # print('hh')
      task = {}
      if not self.eval_only:
        base_class_ids = self.sample_class_ids("base")
        # print(base_class_ids)
        # print(base_class_ids)
        


        #cid: relative. self.class_set[cid]: absolute.
        num_images_per_class = np.array([
          len(self.dataset_spec["images_per_class"][self.base_class_set[cid]]) for cid in base_class_ids
          ])

        if self.num_query_base is not None:
          num_query_base = self.num_query_base
        else:
          num_query_base = compute_num_query(
              num_images_per_class,
              max_num_query=self.max_num_query,
              num_support=self.num_support)
        
        if self.num_support is not None:
          if isinstance(self.num_support, int):
              if any(self.num_support + num_query_base > num_images_per_class):
                  raise ValueError('Some classes do not have enough examples.')
              num_support = self.num_support
          else:
              start, end = self.num_support
              if any(end + num_query_base > num_images_per_class):
                  raise ValueError('The range provided for uniform sampling of the '
                              'number of support examples per class is not valid: '
                              'some classes do not have enough examples.')
              num_support = self._rng.randint(low=start, high=end + 1)
          num_support_per_class = np.array([num_support for _ in base_class_ids])
        else:
          num_remaining_per_class = num_images_per_class - num_query_base
          # print(num_remaining_per_class)

          support_set_size = sample_support_set_size(
              num_remaining_per_class,
              self.max_support_size_contrib_per_class,
              max_support_set_size=self.max_support_set_size,
              rng=self._rng)

          num_support_per_class = sample_num_support_per_class(
              num_images_per_class,
              num_remaining_per_class,
              support_set_size,
              self.max_support_size_contrib_per_class,
              min_log_weight=self.min_log_weight,
              max_log_weight=self.max_log_weight,
              rng=self._rng)  

        total_num_per_class = num_query_base+num_support_per_class

        # class id in the task
        in_task_class_id = 0

        base_images = collections.defaultdict(list)
        base_labels = collections.defaultdict(list)

        for i, cid in enumerate(base_class_ids):


          # random sampling of images.
          all_selected_files = self._rng.choice(self.dataset_spec["images_per_class"][self.base_class_set[cid]],
                                                total_num_per_class[i], False)

          for file_ in all_selected_files[total_num_per_class[i]-num_query_base:]:
              base_images["query"].append(file_)
              base_labels["query"].append(torch.tensor([in_task_class_id]))
          
              
          
          for file_ in all_selected_files[:total_num_per_class[i]-num_query_base]:
              base_images["support"].append(file_)
              base_labels["support"].append(torch.tensor([in_task_class_id]))


          in_task_class_id += 1

        base_labels["query"] = torch.stack(base_labels["query"])
        base_labels["support"] = torch.stack(base_labels["support"])
        base_abs_class_ids = [self.base_class_set[cid] for cid in base_class_ids]
        task["base_images"] = base_images
        task["base_labels"] = base_labels
        task["base_class_ids"] = base_abs_class_ids


      if self.eval_only or self.base2novel:
        
        novel_class_ids = self.sample_class_ids("novel")
        #cid: relative. self.class_set[cid]: absolute.
        num_images_per_class = np.array([
          len(self.dataset_spec["images_per_class"][self.novel_class_set[cid]]) for cid in novel_class_ids
          ])
        num_query_novel_per_class = np.array([self.num_query_novel for _ in novel_class_ids])
        num_query_novel_per_class = np.minimum(num_images_per_class, num_query_novel_per_class)

        
        # class id in the task
        in_task_class_id = 0


        # no support set for novel classes
        novel_images = []
        novel_labels = []

        for i, cid in enumerate(novel_class_ids):


          # random sampling of images.
          all_selected_files = self._rng.choice(self.dataset_spec["images_per_class"][self.novel_class_set[cid]],
                                                num_query_novel_per_class[i], False)

          for file_ in all_selected_files:
              novel_images.append(file_)
              novel_labels.append(torch.tensor([in_task_class_id]))



          in_task_class_id += 1
        
        novel_labels = torch.stack(novel_labels)
        novel_abs_class_ids = [self.novel_class_set[cid] for cid in novel_class_ids]
        task["novel_images"] = novel_images
        task["novel_labels"] = novel_labels
        task["novel_class_ids"] = novel_abs_class_ids
      

      return task

  def sample_multiple_episode(self, batchsize):
      all_tasks = []

      
      for task_index in range(batchsize):
        all_tasks.append(self.sample_single_episode())

      return all_tasks 



class BatchSampler(object):
  """Generates samples of a simple batch.
  In particular, for each batch, it will sample all files and labels of that batch.
  """
  def __init__(self, seed, dataset_spec, split):
    """
    seed: seed for sampling
    dataset_spec: dataset specification
    split: which split to sample from
    """

    # Fixing seed for sampling
    self._rng = np.random.RandomState(seed)



    self.dataset_spec = dataset_spec
    self.split = split

    # all class ids
    if dataset_spec["name"] == "Omniglot":
      self.class_set = get_bilevel_classes(self.split, self.dataset_spec)
    else:
      self.class_set = get_classes(self.split, self.dataset_spec["num_classes_per_split"])
    
    # all files
    self.all_file_path = []
    self.all_labels = []
    
    for class_id in self.class_set:
      self.all_file_path.extend(dataset_spec["images_per_class"][class_id])
      self.all_labels.extend([class_id]*len(dataset_spec["images_per_class"][class_id]))
    self.length = len(self.all_file_path)

    self.init()

  def init(self):
    self.batch_id = 0
    
  def shuffle_data(self):
    indexes = list(range(self.length))

    # random shuffle
    self._rng.shuffle(indexes)
    self.all_file_path = [i for _,i in sorted(zip(indexes,self.all_file_path))]
    self.all_labels = [i for _,i in sorted(zip(indexes,self.all_labels))]



  def sample_batch(self, batch_size, shuffle = True):
    # reset batch_id
    if self.batch_id*batch_size>=self.length:
      self.init()

    # Shuffle the data after completing a round of the dataset
    if shuffle and self.batch_id == 0:
      self.shuffle_data()


    file_paths = self.all_file_path[self.batch_id*batch_size:min(self.length, (self.batch_id+1)*batch_size)]
    labels = torch.tensor(self.all_labels[self.batch_id*batch_size:min(self.length, (self.batch_id+1)*batch_size)])

    images = []
    for file_ in file_paths:
      images.append(file_)

    self.batch_id += 1
    return images, labels


    






