from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from .dataset_spec import Split, create_spec
from .bulid_transforms import build_Torch_transform
import torch
import numpy as np
from .sampling import EpisodeSampler, BatchSampler
import collections

def find_index(random_number, sampling_frequency):
    """
    determine dataset index given sampling frequencies.
    a random float number in [0,1] is used to sample the dataset index
    according to the frequencies.

    args:
        random_number: a float number in [0,1]
        sampling_frequency: a list of float numbers summing to 1.
    """
    index = 0
    sum_ = sampling_frequency[0]
    while sum_ < random_number:
        index += 1
        sum_ += sampling_frequency[index]
    return index


def preprocess(task, transforms):

    all_images = []
    if "base_images" in task:
      processed_base_images = collections.defaultdict(list)
      for path in task["base_images"]["support"]:
          image = Image.open(path).convert('RGB')
        #   print(image.size)
          processed_base_images["support"].append(transforms(image))
      for path in task["base_images"]["query"]:
          image = Image.open(path).convert('RGB')
          processed_base_images["query"].append(transforms(image))
      all_images.append(torch.stack(processed_base_images["support"]))
      all_images.append(task["base_labels"]["support"])
      all_images.append(torch.stack(processed_base_images["query"]))
      all_images.append(task["base_labels"]["query"])
      all_images.append(task["base_class_ids"])
      
    #   processed_base_images["support"] = torch.stack(processed_base_images["support"])
    #   processed_base_images["query"] = torch.stack(processed_base_images["query"])
    #   task["base_images"] = processed_base_images_support

    if "novel_images" in task:
      processed_novel_images = []
      for path in task["novel_images"]:
          image = Image.open(path).convert('RGB')
          processed_novel_images.append(transforms(image))
    #   task["novel_images"] = torch.stack(processed_novel_images)
      all_images.append(torch.stack(processed_novel_images))
      all_images.append(task["novel_labels"])
      all_images.append(task["novel_class_ids"])
      
    return all_images

class TorchDataset(Dataset):
    def __init__(self,
                 dataset_names, 
                 dataset_roots,
                 batch_size, 
                 seed,
                 transforms,
                 is_episodic=True, 
                 episode_descr_config = None,
                 iteration_per_epoch = None,
                 shuffle = False,
                 path_to_words = None, 
                 path_to_is_a = None, 
                 path_to_num_leaf_images=None,
                 train_split_only = False,
                 base2novel = False,
                 cross_dataset = False,
                 ):
        """
        split: which split to sample from
        dataset_names: a list of all dataset names that will be used
        dataset_roots: a list of all dataset roots
        sampling_frequency: a list of float numbers representing sampling
                            frequencies of each dataset.
        batch_size: number of tasks per iteration(for episodic training/test)
                    or batch size
        seed: seed for sampling
        transforms: image transformations
        is_episodic: episodic training/test or not
        episode_descr_config: detailed configurations about how to sample a episode
        iteration_per_epoch: the number of iterations per epoch. Only works for non-episodic training/test.
        shuffle: shuffle a batch or not. Only works for non-episodic training/test.
        path_to_words: path to words.txt.
        path_to_is_a: path to wordnet.is_a.txt
        path_to_num_leaf_images: path to save computed dict mapping the WordNet id 
                                of each ILSVRC 2012 class to its number of images.
        train_split_only: whether use all classes of ImageNet for training.
        """
        
        self._rng = np.random.RandomState(seed)
        self.batch_size = batch_size
        self.is_episodic = is_episodic
        self.shuffle = shuffle
        self.iteration_per_epoch = iteration_per_epoch
        self.episode_descr_config = episode_descr_config
        self.samplers = [] 
        self.transforms = transforms
        self.seed = seed
        self.base2novel = base2novel
        self.cross_dataset = cross_dataset
        
        assert len(dataset_names)==len(dataset_roots)
        
        # construct dataset specifications
        self.dataset_specs = []
        for i, dataset_name in enumerate(dataset_names):
            if dataset_name == "ILSVRC":
                self.dataset_specs.append(create_spec(dataset_name,
                                                 dataset_roots[i],
                                                 path_to_words, 
                                                 path_to_is_a, 
                                                 path_to_num_leaf_images,
                                                 train_split_only))
            else:
                self.dataset_specs.append(create_spec(dataset_name, dataset_roots[i]))

        # construct samplers of each dataset

        assert episode_descr_config is not None

        for i, dataset_spec in enumerate(self.dataset_specs):
            if i ==0 or not self.cross_dataset:
                self.samplers.append(EpisodeSampler(
                    seed,dataset_spec,episode_descr_config,base2novel))
            else:
                self.samplers.append(EpisodeSampler(
                    seed,dataset_spec,episode_descr_config,eval_only=True))
        self.iteration_per_epoch = episode_descr_config.NUM_TASKS_PER_EPOCH//self.batch_size


        # all task/batches of an epoch
        self.all_tasks = []
        self.set_epoch()



    def __len__(self):
        return self.iteration_per_epoch

    def set_epoch(self) -> None:
        # sample all tasks/batches of images
        self.all_tasks = []
        
        # sample tasks/batches
        for _ in range(self.iteration_per_epoch):
            # [0,1] uniform sampling to determine the dataset to sample from

            # dataset index

            whole_task = []
            for sampler in self.samplers:
                whole_task.append(sampler.sample_multiple_episode(self.batch_size))
            whole_task_ = []
            for i in range(self.batch_size):
                one_task = []
                for tasks_per_dataset in whole_task:
                    one_task.append(tasks_per_dataset[i])
                whole_task_.append(one_task)
            self.all_tasks.append(whole_task_)


    
    def __getitem__(self, index):
        # sanple a batch of tasks

        current_tasks = self.all_tasks[index]
        new_tasks = []
        all_images = []
        for one_task in current_tasks:
            new_one_task = []
            for task_per_dataset in one_task:
                new_one_task.append(preprocess(task_per_dataset, self.transforms))
                # new_one_task.append(task_per_dataset)
            new_tasks.append(new_one_task)

        return new_tasks

    




def create_torch_dataloader(config):
    # create a dataloader
    is_train = False
    config_ = config.DATA.TEST    


    transforms = build_Torch_transform(is_train, config)

    path_to_words = config.DATA.PATH_TO_WORDS
    path_to_is_a = config.DATA.PATH_TO_IS_A
    path_to_num_leaf_images = config.DATA.PATH_TO_NUM_LEAF_IMAGES



    dataset = TorchDataset(config_.DATASET_NAMES,
                            config_.DATASET_ROOTS,
                            config_.BATCH_SIZE, 
                            config.SEED,
                            transforms,
                            config_.IS_EPISODIC,
                            config_.EPISODE_DESCR_CONFIG,
                            config_.ITERATION_PER_EPOCH,
                            config_.SHUFFLE,
                            config.DATA.PATH_TO_WORDS,
                            config.DATA.PATH_TO_IS_A,
                            config.DATA.PATH_TO_NUM_LEAF_IMAGES,
                            config.DATA.TRAIN_SPLIT_ONLY,
                            config.DATA.BASE2NOVEL,
                            config.DATA.CROSS_DATASET
                            )
        
    loader = DataLoader(dataset,
                        num_workers = config.DATA.NUM_WORKERS,
                        pin_memory = config.DATA.PIN_MEMORY)
    
    return loader, dataset

    
    
    


