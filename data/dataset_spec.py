"""
Interfaces for dataset specifications.
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



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from collections import abc, OrderedDict
import json
import os

from absl import logging
import numpy as np
import itertools
import enum
from scipy.io import loadmat
from .ImageNet_graph_operations import *
import json
import operator

# The seed is fixed, in order to ensure reproducibility of the split generation,
# exactly matching the original Meta-Dataset code.
SEED = 22

AUX_DATA_PATH = os.path.dirname(os.path.realpath(__file__))
VGGFLOWER_LABELS_PATH = f'{AUX_DATA_PATH}/VggFlower_labels.txt'
TRAFFICSIGN_LABELS_PATH = f'{AUX_DATA_PATH}/TrafficSign_labels.txt'
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

class Split(enum.Enum):
  """The possible data splits."""
  BASE = 0
  NOVEL = 1


def has_file_allowed_extension(filename: str, extensions) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def gen_rand_split_inds(num_base_classes, num_novel_classes, _rng):
  """Generates a random set of indices corresponding to dataset splits.
  It assumes the indices go from [0, num_classes), where the num_classes =
  num_train_classes + num_val_classes + num_test_classes. The returned indices
  are non-overlapping and cover the entire range.
  Note that in the current implementation, valid_inds and test_inds are sorted,
  but train_inds is in random order.
  Args:
    num_base_classes : int, number of base classes.
    num_novel_classes : int, number of novel classes.
    _rng : numpy fixed random number generator, used to match the split used in the 
           original benchmark.
  Returns:
    base_inds : np array of base inds.
    novel_inds : np array of valid inds.
  """
  num_classes = num_base_classes + num_novel_classes

  # First split into trainval and test splits.
  base_inds = _rng.choice(
      num_classes, num_base_classes, replace=False)
  novel_inds = np.setdiff1d(np.arange(num_classes), base_inds)
  print(
      f'Created splits with {len(base_inds)} base, {len(novel_inds)} novel classes.')
  return base_inds.tolist(), novel_inds.tolist()

def gen_sequential_split_inds(num_train_classes, num_valid_classes, num_test_classes):
  """Generates a sequential set of indices corresponding to dataset splits.
  It assumes the indices go from [0, num_classes), where the num_classes =
  num_train_classes + num_val_classes + num_test_classes. The returned indices
  are non-overlapping and cover the entire range.
  Args:
    num_train_classes : int, number of (meta)-training classes.
    num_valid_classes : int, number of (meta)-valid classes.
    num_test_classes : int, number of (meta)-test classes.
  Returns:
    train_inds : np array of training inds.
    valid_inds : np array of valid inds.
    test_inds  : np array of test inds.
  """
  train_inds = list(range(num_train_classes))
  valid_inds = list(range(num_train_classes,num_train_classes+num_valid_classes))
  test_inds = list(range(num_train_classes+num_valid_classes, num_train_classes+num_valid_classes+num_test_classes))
  return train_inds, valid_inds, test_inds

def create_spec(dataset_name, root, path_to_words=None, path_to_is_a = None, path_to_num_leaf_images = None, train_split_only = False, all_test = False):
  """
  create a dataset specification.
  """
  if dataset_name == "Textures":
    return create_DTD_spec(root)
  elif dataset_name == "ILSVRC":
    return create_ImageNet_spec(root, path_to_words, path_to_is_a, path_to_num_leaf_images, train_split_only)
  elif dataset_name == "Quick Draw":
    return create_QuickDraw_spec(root)
  elif dataset_name == "VGG Flower":
    return create_VGGFlower_spec(root)
  elif dataset_name == "Aircraft":
    return create_Aircraft_spec(root)
  elif dataset_name == "Fungi":
    return create_fungi_spec(root)
  elif dataset_name == "CIFAR100":
    return create_cifar100_spec(root)
  elif dataset_name == "euroSAT":
    return create_euroSAT_spec(root)
  elif dataset_name == "sun397":
    return create_sun397_spec(root)
  elif dataset_name == "ucf":
    return create_ucf_spec(root)
  elif dataset_name == "plantD":
    return create_plantD_spec(root)
  elif dataset_name == "CUB":
    return create_CUB_spec(root)



def create_DTD_spec(root):
  """
  Return a dataset specification that includes:
    name: The name of the dataset.
    num_classes_per_split: number of images per split.
    images_per_class: a dictionary containing all paths of images in each class.
    id2name: a dictionary mapping class id to real name.
  """
  NUM_BASE_CLASSES = 37
  NUM_NOVEL_CLASSES = 10
  _rng = np.random.RandomState(SEED)
  base_inds, novel_inds = gen_rand_split_inds(
        NUM_BASE_CLASSES, NUM_NOVEL_CLASSES, _rng)

  

  class_names = sorted(
        os.listdir(os.path.join(root, 'images')))

  splits = {
        Split.BASE: [class_names[i] for i in base_inds],
        Split.NOVEL: [class_names[i] for i in novel_inds]
    }


  dataset_specification = {}
  dataset_specification["name"] = "Textures"
  dataset_specification["num_classes_per_split"] = {
        Split.BASE: len(splits[Split.BASE]),
        Split.NOVEL: len(splits[Split.NOVEL])
    }

  all_classes = list(
        itertools.chain(splits[Split.BASE], splits[Split.NOVEL]))
  class_names = {}

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}

  for class_id, class_name in enumerate(all_classes):
    dataset_specification["id2name"][class_id] = class_name
    dataset_specification["images_per_class"][class_id] = []
    for file_ in os.listdir(os.path.join(root, 'images',class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, 'images',class_name,file_))
    dataset_specification["images_per_class"][class_id].sort()
  return dataset_specification

def create_CUB_spec(root):
  """
  Return a dataset specification that includes:
    name: The name of the dataset.
    num_classes_per_split: number of images per split.
    images_per_class: a dictionary containing all paths of images in each class.
    id2name: a dictionary mapping class id to real name.
  """
  NUM_BASE_CLASSES = 160
  NUM_NOVEL_CLASSES = 40
  NUM_TOTAL_CLASSES = NUM_BASE_CLASSES + NUM_NOVEL_CLASSES

  _rng = np.random.RandomState(SEED)
  base_inds, novel_inds = gen_rand_split_inds(
        NUM_BASE_CLASSES, NUM_NOVEL_CLASSES, _rng)

  with open(os.path.join(root, 'classes.txt'), 'r') as f:
    class_names = []
    for lines in f:
      _, class_name = lines.strip().split(' ')
      class_names.append(class_name)
  
  err_msg = 'number of classes in dataset does not match split specification'
  assert len(class_names) == NUM_TOTAL_CLASSES, err_msg

  splits = {
        Split.BASE: [class_names[i] for i in base_inds],
        Split.NOVEL: [class_names[i] for i in novel_inds]
    }


  dataset_specification = {}
  dataset_specification["name"] = "Birds"
  dataset_specification["num_classes_per_split"] = {
        Split.BASE: len(splits[Split.BASE]),
        Split.NOVEL: len(splits[Split.NOVEL])
    }

  all_classes = list(
        itertools.chain(splits[Split.BASE], splits[Split.NOVEL]))
  class_names = {}

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}

  def get_real_names(dir_name):
    _, dir_name = dir_name.split(".")
    split_names = dir_name.split("_")
    real_name = ""
    for split in split_names:
      real_name += f"{split} "
    real_name = real_name.strip()
    return real_name

  for class_id, class_name in enumerate(all_classes):
    dataset_specification["id2name"][class_id] = get_real_names(class_name)
    dataset_specification["images_per_class"][class_id] = []
    for file_ in os.listdir(os.path.join(root, 'images',class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, 'images',class_name,file_))
    dataset_specification["images_per_class"][class_id].sort()
  return dataset_specification

def create_sun397_spec(root):
  NUM_BASE_CLASSES = 318
  NUM_NOVEL_CLASSES = 79
  NUM_TOTAL_CLASSES = NUM_BASE_CLASSES + NUM_NOVEL_CLASSES

  _rng = np.random.RandomState(SEED)
  base_inds, novel_inds = gen_rand_split_inds(
        NUM_BASE_CLASSES, NUM_NOVEL_CLASSES, _rng)
  


  classname2dir = {}
  with open(os.path.join(root, "ClassName.txt"), "r") as f:
    lines = f.readlines()
    for line in lines:
      line = line.strip()[1:]
      # classname = os.path.dirname(line)
      names = line.split("/")[1:]
      names = names[::-1]
      classname = " ".join(names)
      classname2dir[classname] = os.path.join(root, line)
  

  classnames = list(classname2dir.keys())
  splits = {
        Split.BASE: [classnames[i] for i in base_inds],
        Split.NOVEL: [classnames[i] for i in novel_inds]
    }
  

  dataset_specification = {}
  dataset_specification["name"] = "sun397"

  dataset_specification["num_classes_per_split"] = {
        Split.BASE: len(splits[Split.BASE]),
        Split.NOVEL: len(splits[Split.NOVEL])
    }

  all_classes = list(
        itertools.chain(splits[Split.BASE], splits[Split.NOVEL]))
  class_names = {}

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}

  def get_real_names(dir_name):
    split_names = dir_name.split("_")
    real_name = ""
    for split in split_names:
      real_name += f"{split} "
    real_name = real_name.strip()
    return real_name

  for class_id, class_name in enumerate(all_classes):
    # logging.info('Creating record for class ID %d (%s)...', class_id,
                  #  class_name)
    dataset_specification["id2name"][class_id] = get_real_names(class_name)
    dataset_specification["images_per_class"][class_id] = []
    

    for file_ in os.listdir(classname2dir[class_name]):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(classname2dir[class_name],file_))
    dataset_specification["images_per_class"][class_id].sort()
    # if class_id == 0:
    #   print(dataset_specification["images_per_class"][class_id][:10])
    #   print(dataset_specification["id2name"][class_id])
  # print(dataset_specification["id2name"])
  return dataset_specification

def create_ucf_spec(root):
  NUM_BASE_CLASSES = 81
  NUM_NOVEL_CLASSES = 20
  NUM_TOTAL_CLASSES = NUM_BASE_CLASSES + NUM_NOVEL_CLASSES

  _rng = np.random.RandomState(SEED)
  base_inds, novel_inds = gen_rand_split_inds(
        NUM_BASE_CLASSES, NUM_NOVEL_CLASSES, _rng)

  class_names = sorted(
        os.listdir(root))
  
  splits = {
        Split.BASE: [class_names[i] for i in base_inds],
        Split.NOVEL: [class_names[i] for i in novel_inds]
    }

  dataset_specification = {}
  dataset_specification["name"] = "ucf"

  dataset_specification["num_classes_per_split"] = {
        Split.BASE: len(splits[Split.BASE]),
        Split.NOVEL: len(splits[Split.NOVEL])
    }

  all_classes = list(
        itertools.chain(splits[Split.BASE], splits[Split.NOVEL]))
  class_names = {}

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}

  def get_real_names(dir_name):
    split_names = dir_name.split("_")
    real_name = ""
    for split in split_names:
      real_name += f"{split} "
    real_name = real_name.strip()
    return real_name

  for class_id, class_name in enumerate(all_classes):
    # logging.info('Creating record for class ID %d (%s)...', class_id,
                  #  class_name)
    dataset_specification["id2name"][class_id] = get_real_names(class_name)
    dataset_specification["images_per_class"][class_id] = []
    

    for file_ in os.listdir(os.path.join(root, class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, class_name,file_))
    dataset_specification["images_per_class"][class_id].sort()
  # print(dataset_specification["id2name"])
  return dataset_specification

def create_plantD_spec(root):

  names = {}
  names["Apple___Apple_scab"] = "apple scab of apple leaf"
  names["Apple___Black_rot"] = "black rot of apple leaf"
  names["Apple___Cedar_apple_rust"] = "cedar apple rust leaf"
  names["Apple___healthy"] = "healthy apple leaf"
  names["Blueberry___healthy"] = "healthy blueberry leaf"
  names["Cherry_(including_sour)___healthy"] = "healthy sour cherry leaf"
  names["Cherry_(including_sour)___Powdery_mildew"] = "powdery mildew of sour cherry leaf"
  names["Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"] = "gray leaf spot or cercospora leaf spot of corn leaf"
  names["Corn_(maize)___Common_rust_"] = "common rust of corn leaf"
  names["Corn_(maize)___healthy"] = "healthy corn leaf"
  names["Corn_(maize)___Northern_Leaf_Blight"] = "northern corn leaf blight"
  names["Grape___Black_rot"] = "black rot of grape leaf"
  names["Grape___Esca_(Black_Measles)"] = "esca (black measles) of grape leaf"
  names["Grape___healthy"] = "healthy grape leaf"
  names["Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"] = "leaf blight (Isariopsis leaf spot) of grape leaf"
  names["Orange___Haunglongbing_(Citrus_greening)"] = "citrus greening of orange leaf"
  names["Peach___Bacterial_spot"] = "bacterial spot of peach leaf"
  names["Peach___healthy"] = "healthy bell peach leaf"
  names["Pepper,_bell___Bacterial_spot"] = "bacterial spot of bell pepper leaf"
  names["Pepper,_bell___healthy"] = "healthy bell pepper leaf"
  names["Potato___Early_blight"] = "early blight of potato leaf"
  names["Potato___healthy"] = "healthy potato leaf"
  names["Potato___Late_blight"] = "late blight of potato leaf"
  names["Raspberry___healthy"] = "healthy raspberry leaf"
  names["Soybean___healthy"] = "healthy soybean leaf"
  names["Squash___Powdery_mildew"] = "powdery mildew of squash leaf"
  names["Strawberry___healthy"] = "healthy strawberry leaf"
  names["Strawberry___Leaf_scorch"] = "leaf scorch of strawberry leaf"
  names["Tomato___Bacterial_spot"] = "bacterial spot of tomato leaf"
  names["Tomato___Early_blight"] = "early blight of tomato leaf"
  names["Tomato___healthy"] = "healthy tomato leaf"
  names["Tomato___Late_blight"] = "late blight of tomato leaf"
  names["Tomato___Leaf_Mold"] = "leaf mold of tomato leaf"
  names["Tomato___Septoria_leaf_spot"] = "septoria leaf spot of tomato leaf"
  names["Tomato___Spider_mites Two-spotted_spider_mite"] = "tomato leaf with (twospotted) spider mites"
  names["Tomato___Target_Spot"] = "target spot of tomato leaf"
  names["Tomato___Tomato_mosaic_virus"] = "tomato mosaic virus of tomato leaf"
  names["Tomato___Tomato_Yellow_Leaf_Curl_Virus"] = "tomato yellow leaf curl virus of tomato leaf"



  NUM_BASE_CLASSES = 30
  NUM_NOVEL_CLASSES = 8
  NUM_TOTAL_CLASSES = NUM_BASE_CLASSES + NUM_NOVEL_CLASSES

  _rng = np.random.RandomState(SEED)
  base_inds, novel_inds = gen_rand_split_inds(
        NUM_BASE_CLASSES, NUM_NOVEL_CLASSES, _rng)

  class_names = sorted(
        os.listdir(root))
  
  splits = {
        Split.BASE: [class_names[i] for i in base_inds],
        Split.NOVEL: [class_names[i] for i in novel_inds]
    }

  dataset_specification = {}
  dataset_specification["name"] = "plantD"

  dataset_specification["num_classes_per_split"] = {
        Split.BASE: len(splits[Split.BASE]),
        Split.NOVEL: len(splits[Split.NOVEL])
    }

  all_classes = list(
        itertools.chain(splits[Split.BASE], splits[Split.NOVEL]))
  class_names = {}

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}



  for class_id, class_name in enumerate(all_classes):
    # logging.info('Creating record for class ID %d (%s)...', class_id,
                  #  class_name)
    dataset_specification["id2name"][class_id] = names[class_name]
    dataset_specification["images_per_class"][class_id] = []
    

    for file_ in os.listdir(os.path.join(root, class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, class_name,file_))
    dataset_specification["images_per_class"][class_id].sort()
  # print(dataset_specification["id2name"])
  return dataset_specification
    

def create_QuickDraw_spec(root):
  num_classes = 345

  NUM_BASE_CLASSES = 276
  NUM_NOVEL_CLASSES = num_classes - NUM_BASE_CLASSES

  _rng = np.random.RandomState(SEED)
  base_inds, novel_inds = gen_rand_split_inds(
        NUM_BASE_CLASSES, NUM_NOVEL_CLASSES, _rng)

  class_names = sorted(
        os.listdir(root))
  
  splits = {
        Split.BASE: [class_names[i] for i in base_inds],
        Split.NOVEL: [class_names[i] for i in novel_inds]
    }

  dataset_specification = {}
  dataset_specification["name"] = "Quick Draw"

  dataset_specification["num_classes_per_split"] = {
        Split.BASE: len(splits[Split.BASE]),
        Split.NOVEL: len(splits[Split.NOVEL])
    }

  all_classes = list(
        itertools.chain(splits[Split.BASE], splits[Split.NOVEL]))
  class_names = {}

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}

  def get_real_names(dir_name):
    split_names = dir_name.split("_")
    real_name = ""
    for split in split_names:
      real_name += f"{split} "
    real_name = real_name.strip()
    return real_name

  for class_id, class_name in enumerate(all_classes):
    # logging.info('Creating record for class ID %d (%s)...', class_id,
                  #  class_name)
    dataset_specification["id2name"][class_id] = get_real_names(class_name)
    dataset_specification["images_per_class"][class_id] = []
    

    for file_ in os.listdir(os.path.join(root, class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, class_name,file_))
    dataset_specification["images_per_class"][class_id].sort()
  # print(dataset_specification["id2name"])
  return dataset_specification

def create_euroSAT_spec(root):
  NEW_CNAMES = {
      "AnnualCrop": "Annual Crop Land",
      "Forest": "Forest",
      "HerbaceousVegetation": "Herbaceous Vegetation Land",
      "Highway": "Highway or Road",
      "Industrial": "Industrial Buildings",
      "Pasture": "Pasture Land",
      "PermanentCrop": "Permanent Crop Land",
      "Residential": "Residential Buildings",
      "River": "River",
      "SeaLake": "Sea or Lake",
  }

  num_classes = len(NEW_CNAMES)

  NUM_BASE_CLASSES = 6
  NUM_NOVEL_CLASSES = num_classes - NUM_BASE_CLASSES

  _rng = np.random.RandomState(SEED)
  base_inds, novel_inds = gen_rand_split_inds(
        NUM_BASE_CLASSES, NUM_NOVEL_CLASSES, _rng)

  class_names = sorted(
        os.listdir(root))
  
  splits = {
        Split.BASE: [class_names[i] for i in base_inds],
        Split.NOVEL: [class_names[i] for i in novel_inds]
    }

  dataset_specification = {}
  dataset_specification["name"] = "euroSAT"

  dataset_specification["num_classes_per_split"] = {
        Split.BASE: len(splits[Split.BASE]),
        Split.NOVEL: len(splits[Split.NOVEL])
    }

  all_classes = list(
        itertools.chain(splits[Split.BASE], splits[Split.NOVEL]))
  class_names = {}

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}


  for class_id, class_name in enumerate(all_classes):
    # logging.info('Creating record for class ID %d (%s)...', class_id,
                  #  class_name)
    dataset_specification["id2name"][class_id] = NEW_CNAMES[class_name]
    dataset_specification["images_per_class"][class_id] = []
    

    for file_ in os.listdir(os.path.join(root, class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, class_name,file_))
    dataset_specification["images_per_class"][class_id].sort()
  # print(dataset_specification["id2name"])
  return dataset_specification

def create_food_spec(root):
  num_classes = 101

  NUM_BASE_CLASSES = 81
  NUM_NOVEL_CLASSES = num_classes - NUM_BASE_CLASSES

  _rng = np.random.RandomState(SEED)
  base_inds, novel_inds = gen_rand_split_inds(
        NUM_BASE_CLASSES, NUM_NOVEL_CLASSES, _rng)

  class_names = sorted(
        os.listdir(root))
  
  splits = {
        Split.BASE: [class_names[i] for i in base_inds],
        Split.NOVEL: [class_names[i] for i in novel_inds]
    }

  dataset_specification = {}
  dataset_specification["name"] = "food"

  dataset_specification["num_classes_per_split"] = {
        Split.BASE: len(splits[Split.BASE]),
        Split.NOVEL: len(splits[Split.NOVEL])
    }

  all_classes = list(
        itertools.chain(splits[Split.BASE], splits[Split.NOVEL]))
  class_names = {}

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}

  def get_real_names(dir_name):
    split_names = dir_name.split("_")
    real_name = ""
    for split in split_names:
      real_name += f"{split} "
    real_name = real_name.strip()
    return real_name

  for class_id, class_name in enumerate(all_classes):
    # logging.info('Creating record for class ID %d (%s)...', class_id,
                  #  class_name)
    dataset_specification["id2name"][class_id] = get_real_names(class_name)
    dataset_specification["images_per_class"][class_id] = []
    

    for file_ in os.listdir(os.path.join(root, class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, class_name,file_))
    dataset_specification["images_per_class"][class_id].sort()
  # print(dataset_specification["id2name"])
  return dataset_specification


def create_VGGFlower_spec(root):
  """
  Return a dataset specification that includes:
    name: The name of the dataset.
    num_classes_per_split: number of images per split.
    images_per_class: a dictionary containing all paths of images in each class.
    id2name: a dictionary mapping class id to real name.
  """
  # There are 102 classes in the VGG Flower dataset. A 70% / 15% / 15% split
  # between train, validation and test maps to roughly 71 / 15 / 16 classes,
  # respectively.
  NUM_BASE_CLASSES = 82
  NUM_NOVEL_CLASSES = 20
  NUM_TOTAL_CLASSES = NUM_BASE_CLASSES + NUM_NOVEL_CLASSES
  ID_LEN = 3

  _rng = np.random.RandomState(SEED)
  base_inds, novel_inds = gen_rand_split_inds(
        NUM_BASE_CLASSES, NUM_NOVEL_CLASSES, _rng)

  # Load class names from the text file
  file_path = VGGFLOWER_LABELS_PATH
  with open(file_path) as fd:
    all_lines = fd.read()

  # First line is expected to be a comment.
  class_names = all_lines.splitlines()[1:]
  # print(class_names)
  err_msg = 'number of classes in dataset does not match split specification'
  assert len(class_names) == NUM_TOTAL_CLASSES, err_msg

  # Provided class labels are numbers started at 1.
  format_str = '%%0%dd.%%s' % ID_LEN
  splits = {
      Split.BASE: [format_str % (i + 1, class_names[i]) for i in base_inds],
      Split.NOVEL: [format_str % (i + 1, class_names[i]) for i in novel_inds]
  }


  imagelabels_path = os.path.join(root, 'imagelabels.mat')
  with open(imagelabels_path, 'rb') as f:
    labels = loadmat(f)['labels'][0]
  filepaths = collections.defaultdict(list)
  for i, label in enumerate(labels):
    filepaths[label].append(
        os.path.join(root, 'jpg', 'image_{:05d}.jpg'.format(i + 1)))


  dataset_specification = {}
  dataset_specification["name"] = "VGG Flower"
  dataset_specification["num_classes_per_split"] = {
        Split.BASE: len(splits[Split.BASE]),
        Split.NOVEL: len(splits[Split.NOVEL])
    }

  all_classes = list(
        itertools.chain(splits[Split.BASE], splits[Split.NOVEL]))


  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}

  def get_real_names(class_label):
    _, class_label = class_label.split(".")
    return class_label

  for class_id, class_label in enumerate(all_classes):
    # We encode the original ID's in the label.
    original_id = int(class_label[:ID_LEN])

    dataset_specification["id2name"][class_id] = get_real_names(class_label)
    dataset_specification["images_per_class"][class_id] = filepaths[original_id]
  return dataset_specification

def create_Aircraft_spec(root):
  """
  Return a dataset specification that includes:
    name: The name of the dataset.
    num_classes_per_split: number of images per split.
    images_per_class: a dictionary containing all paths of images in each class.
    id2name: a dictionary mapping class id to real name.
  """
  # There are 100 classes in the Aircraft dataset. A 70% / 15% / 15%
  # split between train, validation and test maps to 70 / 15 / 15
  # classes, respectively.


  NUM_BASE_CLASSES = 80
  NUM_NOVEL_CLASSES = 20


  _rng = np.random.RandomState(SEED)
  base_inds, novel_inds= gen_rand_split_inds(
        NUM_BASE_CLASSES, NUM_NOVEL_CLASSES, _rng)

  # Sort the class names, for reproducibility.
  class_names = sorted(
      os.listdir(root))

  assert len(class_names) == (
        NUM_BASE_CLASSES + NUM_NOVEL_CLASSES)

  splits = {
        Split.BASE: [class_names[i] for i in base_inds],
        Split.NOVEL: [class_names[i] for i in novel_inds]
    }

  dataset_specification = {}
  dataset_specification["name"] = "Aircraft"
  dataset_specification["num_classes_per_split"] = {
        Split.BASE: len(splits[Split.BASE]),
        Split.NOVEL: len(splits[Split.NOVEL])
    }
  all_classes = list(
        itertools.chain(splits[Split.BASE], splits[Split.NOVEL]))

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}


  for class_id, class_name in enumerate(all_classes):
    dataset_specification["id2name"][class_id] = class_name
    dataset_specification["images_per_class"][class_id] = []
    for file_ in os.listdir(os.path.join(root, class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, class_name,file_))
    dataset_specification["images_per_class"][class_id].sort()


  return dataset_specification

def create_Traffic_spec(root):
  # There are 43 classes in the Traffic Sign dataset, all of which are used for
  # test episodes.
  NUM_TRAIN_CLASSES = 0
  NUM_VALID_CLASSES = 0
  NUM_TEST_CLASSES = 43
  NUM_TOTAL_CLASSES = NUM_TRAIN_CLASSES + NUM_VALID_CLASSES + NUM_TEST_CLASSES

  class_names = os.listdir(root)
  class_number = []

  for class_ in class_names:
    _, number = class_.split("_")
    class_number.append(int(number))

  class_number.sort()
  class_names = [f"class_{number}" for number in class_number]

  splits = {
        Split.TRAIN: [],
        Split.VALID: [],
        Split.TEST: class_names
    }

  dataset_specification = {}
  dataset_specification["name"] = "Traffic Signs"
  dataset_specification["num_classes_per_split"] = {
        Split.TRAIN: len(splits[Split.TRAIN]),
        Split.VALID: len(splits[Split.VALID]),
        Split.TEST: len(splits[Split.TEST])
    }

  all_classes = list(
        itertools.chain(splits[Split.TRAIN], splits[Split.VALID], splits[Split.TEST]))

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}


  for class_id, class_name in enumerate(all_classes):
    dataset_specification["id2name"][class_id] = class_name
    dataset_specification["images_per_class"][class_id] = []
    
    for file_ in os.listdir(os.path.join(root, class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, class_name,file_))
    dataset_specification["images_per_class"][class_id].sort()
    # print(len(dataset_specification["images_per_class"][class_id]))
  # import pdb
  # pdb.set_trace()

  return dataset_specification


def create_fungi_spec(root):
  """
  Return a dataset specification that includes:
    name: The name of the dataset.
    num_classes_per_split: number of images per split.
    images_per_class: a dictionary containing all paths of images in each class.
    id2name: a dictionary mapping class id to real name.
  """
  NUM_BASE_CLASSES = 1115
  NUM_NOVEL_CLASSES = 279

  _rng = np.random.RandomState(SEED)
  base_inds, novel_inds = gen_rand_split_inds(
        NUM_BASE_CLASSES, NUM_NOVEL_CLASSES, _rng)
  # We ignore the original train and validation splits (the test set cannot be
  # used since it is not labeled).
  with open(os.path.join(root, 'train.json')) as f:
    original_train = json.load(f)
  with open(os.path.join(root, 'val.json')) as f:
    original_val = json.load(f)
  
  # The categories (classes) for train and validation should be the same.
  assert original_train['categories'] == original_val['categories']
  # Sort by category ID for reproducibility.
  categories = sorted(
      original_train['categories'], key=operator.itemgetter('id'))

  # Assert contiguous range [0:category_number]
  assert ([category['id'] for category in categories
          ] == list(range(len(categories))))

  # Some categories share the same name (see
  # https://github.com/visipedia/fgvcx_fungi_comp/issues/1)
  # so we include the category id in the label.
  # print(categories)
  labels = [
      '{:04d}.{}'.format(category['id'], category['name'])
      for category in categories
  ]

  # print(labels)


  splits = {
        Split.BASE: [labels[i] for i in base_inds],
        Split.NOVEL: [labels[i] for i in novel_inds]
    }
  
  image_list = original_train['images'] + original_val['images']
  image_id_dict = {}
  for image in image_list:
    # assert this image_id was not previously added
    assert image['id'] not in image_id_dict
    image_id_dict[image['id']] = image

  # Add a class annotation to every image in image_id_dict.
  annotations = original_train['annotations'] + original_val['annotations']
  for annotation in annotations:
    # assert this images_id was not previously annotated
    assert 'class' not in image_id_dict[annotation['image_id']]
    image_id_dict[annotation['image_id']]['class'] = annotation['category_id']

  # dict where the class is the key.
  class_filepaths = collections.defaultdict(list)
  for image in image_list:
    class_filepaths[image['class']].append(
        os.path.join(root, image['file_name']))

  dataset_specification = {}
  dataset_specification["name"] = "Fungi"
  dataset_specification["num_classes_per_split"] = {
        Split.BASE: len(splits[Split.BASE]),
        Split.NOVEL: len(splits[Split.NOVEL])
    }

  all_classes = list(
        itertools.chain(splits[Split.BASE], splits[Split.NOVEL]))

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}



  for class_id, class_label in enumerate(all_classes):
    # Extract the "category_id" information from the class label
    category_id = int(class_label[:4])
    # Check that the key is actually in `class_filepaths`, so that an empty
    # list is not accidentally used.
    if category_id not in class_filepaths:
      raise ValueError('class_filepaths does not contain paths to any '
                        'image for category %d. Existing categories are: %s.' %
                        (category_id, class_filepaths.keys()))
    class_paths = class_filepaths[category_id]
    dataset_specification["id2name"][class_id] = class_label[5:]
    dataset_specification["images_per_class"][class_id] = class_paths
  return dataset_specification


  

def create_ImageNet_spec(root, 
                         path_to_words = None, 
                         path_to_is_a = None, 
                         path_to_num_leaf_images=None,
                         train_split_only = False
                         ):
  """
  Args:
    root: path to ImageNet training set.
    path_to_words: path to words.txt.
    path_to_is_a: path to wordnet.is_a.txt
    path_to_num_leaf_images: path to save computed dict mapping the WordNet id 
                             of each ILSVRC 2012 class to its number of images.
    train_split_only: whether use all classes of ImageNet for training.

  Return a dataset specification that includes:
    name: The name of the dataset.
    num_classes_per_split: number of images per split.
    images_per_class: a dictionary containing all paths of images in each class.
    id2name: a dictionary mapping class id to real name.
    split_subgraph: a dict mapping each split to the set of Synsets in the
                    subgraph of that split.
    class_names_to_ids: a dictionary mapping real name to class id.

  """

  synsets = {}
  if not path_to_words:
    path_to_words = os.path.join(root, 'words.txt')

  with open(path_to_words) as f:
    for line in f:
      wn_id, words = line.rstrip().split('\t')
      synsets[wn_id] = Synset(wn_id, words, set(), set())
  
  # Populate the parents / children arrays of these Synsets.
  if not path_to_is_a:
    path_to_is_a = os.path.join(root, 'wordnet.is_a.txt')

  with open(path_to_is_a, 'r') as f:
    for line in f:
      parent, child = line.rstrip().split(' ')
      synsets[parent].children.add(synsets[child])
      synsets[child].parents.add(synsets[parent])

  wn_ids_2012 = os.listdir(root)
  wn_ids_2012 = set(
    entry for entry in wn_ids_2012
    if os.path.isdir(os.path.join(root, entry)))

  # all leaves in ImageNet
  synsets_2012 = [s for s in synsets.values() if s.wn_id in wn_ids_2012]
  assert len(wn_ids_2012) == len(synsets_2012)

  # Get a dict mapping each WordNet id of ILSVRC 2012 to its number of images.
  num_synset_2012_images = get_num_synset_2012_images(root, path_to_num_leaf_images,
                                                      synsets_2012)

  # Get the graph of all and only the ancestors of the ILSVRC 2012 classes.
  sampling_graph = create_sampling_graph(synsets_2012)

  # Create a dict mapping each node to its reachable leaves.
  spanning_leaves = get_spanning_leaves(sampling_graph)

  # Create a dict mapping each node in sampling graph to the number of images of
  # ILSVRC 2012 synsets that live in the sub-graph rooted at that node.
  num_images = get_num_spanning_images(spanning_leaves, num_synset_2012_images)

  if train_split_only:
    # We are keeping all graph for training.
    valid_test_roots = None
    splits = {
        Split.BASE: spanning_leaves,
        Split.NOVEL: set()
    }
  else:
    # Create class splits, each with its own sampling graph.
    # Choose roots for the validation and test subtrees (see the docstring of
    # create_splits for more information on how these are used).
    novel_roots = {
        'valid': get_synset_by_wnid('n02075296', sampling_graph),  # 'carnivore'
        'test':
            get_synset_by_wnid('n03183080', sampling_graph)  # 'device'
    }
    # The novel_roots returned here correspond to the same Synsets as in
    # the above dict, but are the copied versions of them for each subgraph.
    splits, novel_roots = create_splits(
        spanning_leaves, Split, valid_test_roots=novel_roots)


  # Compute num_images for each split.
  split_num_images = {}
  split_num_images[Split.BASE] = get_num_spanning_images(
      get_spanning_leaves(splits[Split.BASE]), num_synset_2012_images)

  split_num_images[Split.NOVEL] = get_num_spanning_images(
      get_spanning_leaves(splits[Split.NOVEL]), num_synset_2012_images)

  

  # Get a list of synset id's assigned to each split.
  def _get_synset_ids(split):
    """Returns a list of synset id's of the classes assigned to split."""
    return sorted([
        synset.wn_id for synset in get_leaves(
            splits[split])
    ])
  base_synset_ids = _get_synset_ids(Split.BASE)
  novel_synset_ids = _get_synset_ids(Split.NOVEL)

  # print(len(base_synset_ids))
  # print(len(novel_synset_ids))
  all_synset_ids = base_synset_ids + novel_synset_ids

  # By construction of all_synset_ids, we are guaranteed to get train synsets
  # before validation synsets, and validation synsets before test synsets.
  # Therefore the assigned class_labels will respect that partial order.
  class_names = {}
  for class_label, synset_id in enumerate(all_synset_ids):
      class_names[class_label] = synset_id

  dataset_specification = {}
  dataset_specification["name"] = "ILSVRC"
  dataset_specification["split_subgraph"] = splits
  dataset_specification["class_names_to_ids"] = dict(
    zip(class_names.values(), class_names.keys()))
  dataset_specification["id2name"] = {}
  dataset_specification["num_classes_per_split"] = {
        Split.BASE: len(base_synset_ids),
        Split.NOVEL: len(novel_synset_ids)
    }


  def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        ImageNet_classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                ImageNet_classnames[folder] = classname
        return ImageNet_classnames

  ImageNet_classnames = read_classnames("/home/luoxu/data2/classnames.txt")
  
  dataset_specification["images_per_class"] = {}

  for class_id, class_name in class_names.items():
    dataset_specification["id2name"][class_id] = ImageNet_classnames[class_name]
    dataset_specification["images_per_class"][class_id] = []
    for file_ in os.listdir(os.path.join(root, class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, class_name,file_))
    dataset_specification["images_per_class"][class_id].sort()
  return dataset_specification


def create_cifar100_spec(root):
  num_classes = 100

  NUM_BASE_CLASSES = 80
  NUM_NOVEL_CLASSES = num_classes - NUM_BASE_CLASSES

  _rng = np.random.RandomState(SEED)
  base_inds, novel_inds = gen_rand_split_inds(
        NUM_BASE_CLASSES, NUM_NOVEL_CLASSES, _rng)

  class_names = sorted(
        os.listdir(root))
  
  splits = {
        Split.BASE: [class_names[i] for i in base_inds],
        Split.NOVEL: [class_names[i] for i in novel_inds]
    }

  dataset_specification = {}
  dataset_specification["name"] = "cifar100"

  dataset_specification["num_classes_per_split"] = {
        Split.BASE: len(splits[Split.BASE]),
        Split.NOVEL: len(splits[Split.NOVEL])
    }

  all_classes = list(
        itertools.chain(splits[Split.BASE], splits[Split.NOVEL]))
  class_names = {}

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}


  for class_id, class_name in enumerate(all_classes):
    # logging.info('Creating record for class ID %d (%s)...', class_id,
                  #  class_name)
    dataset_specification["id2name"][class_id] = class_name
    dataset_specification["images_per_class"][class_id] = []
    

    for file_ in os.listdir(os.path.join(root, class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, class_name,file_))
    dataset_specification["images_per_class"][class_id].sort()
  # print(dataset_specification["id2name"])

  return dataset_specification






def get_classes(split, classes_per_split):
  """Gets the sequence of class labels for a split.
  Class id's are returned ordered and without gaps.
  Args:
    split: A Split, the split for which to get classes.
    classes_per_split: Matches each Split to the number of its classes.
  Returns:
    The sequence of classes for the split.
  Raises:
    ValueError: An invalid split was specified.
  """

  num_classes = classes_per_split[split]

  # Find the starting index of classes for the given split.
  if split == Split.BASE:
    offset = 0
  elif split == Split.NOVEL:
    offset = classes_per_split[Split.BASE]
  else:
    raise ValueError('Invalid dataset split.')

  # Get a contiguous range of classes from split.
  return range(offset, offset + num_classes)




def get_total_images_per_class(data_spec, class_id=None, pool=None):
  """Returns the total number of images of a class in a data_spec and pool.
  Args:
    data_spec: A DatasetSpecification, or BiLevelDatasetSpecification.
    class_id: The class whose number of images will be returned. If this is
      None, it is assumed that the dataset has the same number of images for
      each class.
    pool: A string ('train' or 'test', optional) indicating which example-level
      split to select, if the current dataset has them.
  Raises:
    ValueError: when
      - no class_id specified and yet there is class imbalance, or
      - no pool specified when there are example-level splits, or
      - pool is specified but there are no example-level splits, or
      - incorrect value for pool.
    RuntimeError: the DatasetSpecification is out of date (missing info).
  """
  if class_id is None:
    if len(set(data_spec.images_per_class.values())) != 1:
      raise ValueError('Not specifying class_id is okay only when all classes'
                       ' have the same number of images')
    class_id = 0

  if class_id not in data_spec.images_per_class:
    raise RuntimeError('The DatasetSpecification should be regenerated, as '
                       'it does not have a non-default value for class_id {} '
                       'in images_per_class.'.format(class_id))
  num_images = data_spec.images_per_class[class_id]

  if pool is None:
    if isinstance(num_images, abc.Mapping):
      raise ValueError('DatasetSpecification {} has example-level splits, so '
                       'the "pool" argument has to be set (to "train" or '
                       '"test".'.format(data_spec.name))
  elif not data.POOL_SUPPORTED:
    raise NotImplementedError('Example-level splits or pools not supported.')

  return num_images


