import logging

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import DatasetFolder
from torchvision.datasets import MNIST, CIFAR10
import os
import torch
import pandas as pd
from torchvision.transforms import ToTensor

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            data = mnist_dataobj.train_data
            target = mnist_dataobj.train_labels
        else:
            data = mnist_dataobj.test_data
            target = mnist_dataobj.test_labels

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

def load_data():
      directory = 'SIDD'
      uid = 0
      imgs = {}
      client_id = 0
      """
      label_counts = {0: 0, 1: 0}

      for client in os.listdir(directory):
          curr_path = f'{directory}/{client}/pcap'

          for subdir in os.listdir(curr_path):
              curr_path = f'{directory}/{client}/pcap/{subdir}/dataset'
              curr_type = subdir[-1:]
              if curr_type == str(1):
                  for dayscen in os.listdir(curr_path):
                      curr_path = f'{directory}/{client}/pcap/{subdir}/dataset/{dayscen}'
                      for i, img in enumerate(os.listdir(curr_path)):
                          if i == 10:
                            break
                          if dayscen == 'benign':
                              label = 0
                              imgs[uid] = {'id': uid, 'client_id': client_id, 'label': str(label), 'fn': img, 'path': curr_path + '/' + img}
                              uid +=1
                              label_counts[label] += 1
                          elif dayscen == 'malicious':
                              label = 1
                              imgs[uid] = {'id': uid, 'client_id': client_id, 'label': str(label), 'fn': img, 'path': curr_path + '/' + img}
                              #imgs[uid+1] = {'id': uid, 'client_id': client_id, 'label': str(label), 'fn': img, 'path': curr_path + '/' + img}
                              #imgs[uid+2] = {'id': uid, 'client_id': client_id, 'label': str(label), 'fn': img, 'path': curr_path + '/' + img}
                              uid += 1
                              label_counts[label] += 1
          client_id += 1
    """
      label_counts = {0: 0, 1: 0}

      for client in os.listdir(directory):
          curr_path = f'{directory}/{client}/pcap'

          for subdir in os.listdir(curr_path):
              
              curr_path = f'{directory}/{client}/pcap/{subdir}/dataset'
              curr_type = subdir[-1:]
              if curr_type == str(1):
                  for dayscen in os.listdir(curr_path):
                      curr_path = f'{directory}/{client}/pcap/{subdir}/dataset/{dayscen}'
                      for i, img in enumerate(os.listdir(curr_path)):
                          if i == 5:
                            break
                          if dayscen == 'benign':
                              label = 0
                          elif dayscen == 'malicious':
                              label = 1
                          imgs[uid] = {'id': uid, 'label': str(label), 'fn': img, 'path': curr_path + '/' + img}
                          uid += 1
                          label_counts[label] += 1
                          
      print(label_counts)
      #print(label_counts[1] / label_counts[0] * 100)

      total = label_counts[1] + label_counts[0]
      neg = label_counts[0]
      pos = label_counts[1]

      weight_for_0 = (1 / neg) * (total / 2.0)
      weight_for_1 = (1 / pos) * (total / 2.0)


      class_weight = {0: weight_for_0, 1: weight_for_1}

      #print('Weight for class 0: {:.2f}'.format(weight_for_0))
      #print('Weight for class 1: {:.2f}'.format(weight_for_1))
      img_df = pd.DataFrame.from_dict(imgs,orient='index')
      #img_df = img_df[img_df['client_id'] == clientID]
      img_df['label'] = img_df['label'].astype(int)
      #img_df['label'] = img_df['label'].replace(3,2)
      #img_df.loc[img_df.index[(img_df['label']==3)],'label'] = 2
      logging.info("Created data frame, length: %s", len(img_df.index))

      return img_df

"""    
def _parse_function(filename, label):
        with open(filename, 'rb') as f:
            image = Image.open(f)
            image = image.convert('L')  # convert to grayscale
            image = torch.tensor(np.array(image), dtype=torch.float32)
            image = image.unsqueeze(0)  # add channel dimension as the first dimension
        return image, label
"""
def _parse_function(filename, label):
    image = Image.open(filename).convert('L')
    image = ToTensor()(image)
    return image, label


class SIDD(data.Dataset):
    #def __init__(self, root, dataidxs=None):
    def __init__(self, dataidxs=None):

        self.dataidxs = dataidxs

        img_df = load_data()
        file_paths = img_df.path
        file_labels = img_df["label"]

        self.ds_length = len(img_df)
        self.file_paths = file_paths
        self.file_labels = file_labels

    def __getitem__(self, idx):

        filename = self.file_paths[idx]
        label = self.file_labels[idx]
        image, label = _parse_function(filename, label)
        return image, label


    def __len__(self):
        return self.ds_length

class SIDD_truncated(data.Dataset):
    def __init__(self, dataidxs=None):
        super().__init__()

        self.dataidxs = dataidxs
        self.sidd = SIDD()
        
        if self.dataidxs is not None:
            self.data = [self.sidd.file_paths[i] for i in self.dataidxs]
            self.target = [self.sidd.file_labels[i] for i in self.dataidxs]
        else:
            self.data = self.sidd.file_paths
            self.target = self.sidd.file_labels

    def __getitem__(self, idx):
        filename = self.data[idx]
        label = self.target[idx]
        image, label = _parse_function(filename, label)
        return image, label

    def __len__(self):
        return len(self.data)

class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            #print("train member of the class: {}".format(self.train))
            #data = cifar_dataobj.train_data
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)



class CIFAR10ColorGrayScale(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform_color=None, transofrm_gray_scale=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform_color = transform_color
        self.transofrm_gray_scale = transofrm_gray_scale
        self.target_transform = target_transform
        self.download = download
        self._gray_scale_indices = []

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, None, self.target_transform, self.download)

        if self.train:
            #print("train member of the class: {}".format(self.train))
            #data = cifar_dataobj.train_data
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        self._gray_scale_indices = index
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = self.data[gs_index, :, :, 0]
            self.data[gs_index, :, :, 2] = self.data[gs_index, :, :, 0]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        #if self.transform is not None:
        if index in self._gray_scale_indices:
            if self.transofrm_gray_scale is not None:
                img = self.transofrm_gray_scale(img)
        else:
            if self.transform_color is not None:
                img = self.transform_color(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)



class CIFAR10ColorGrayScaleTruncated(data.Dataset):
    def __init__(self, root, dataidxs=None, gray_scale_indices=None,
                    train=True, transform_color=None, transofrm_gray_scale=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform_color = transform_color
        self.transofrm_gray_scale = transofrm_gray_scale
        self.target_transform = target_transform
        self._gray_scale_indices = gray_scale_indices
        self.download = download

        self.cifar_dataobj = CIFAR10(self.root, self.train, None, self.target_transform, self.download)

        # we need to trunc the channle first
        self.__truncate_channel__(index=gray_scale_indices)
        # then we trunct he dataset
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.train:
            data = self.cifar_dataobj.data
            target = np.array(self.cifar_dataobj.targets)
        else:
            data = self.cifar_dataobj.data
            target = np.array(self.cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __truncate_channel__(self, index):
        #self._gray_scale_indices = index
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.cifar_dataobj.data[gs_index, :, :, 1] = self.cifar_dataobj.data[gs_index, :, :, 0]
            self.cifar_dataobj.data[gs_index, :, :, 2] = self.cifar_dataobj.data[gs_index, :, :, 0]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        #if self.transform is not None:
        if index in self._gray_scale_indices:
            if self.transofrm_gray_scale is not None:
                img = self.transofrm_gray_scale(img)
        else:
            if self.transform_color is not None:
                img = self.transform_color(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR10ColorGrayScaleOverSampled(data.Dataset):
    '''
    Here we conduct oversampling strategy (over the underrepresented domain) in mitigating the data bias
    '''
    def __init__(self, root, dataidxs=None, gray_scale_indices=None,
                    train=True, transform_color=None, transofrm_gray_scale=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform_color = transform_color
        self.transofrm_gray_scale = transofrm_gray_scale
        self.target_transform = target_transform
        self._gray_scale_indices = gray_scale_indices
        self.download = download

        self.cifar_dataobj = CIFAR10(self.root, self.train, None, self.target_transform, self.download)

        # we need to trunc the channle first
        self.__truncate_channel__(index=gray_scale_indices)
        # then we trunct he dataset
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.train:
            data = self.cifar_dataobj.data
            target = np.array(self.cifar_dataobj.targets)
        else:
            data = self.cifar_dataobj.data
            target = np.array(self.cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __truncate_channel__(self, index):
        #self._gray_scale_indices = index
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.cifar_dataobj.data[gs_index, :, :, 1] = self.cifar_dataobj.data[gs_index, :, :, 0]
            self.cifar_dataobj.data[gs_index, :, :, 2] = self.cifar_dataobj.data[gs_index, :, :, 0]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        #if self.transform is not None:
        if index in self._gray_scale_indices:
            if self.transofrm_gray_scale is not None:
                img = self.transofrm_gray_scale(img)
        else:
            if self.transform_color is not None:
                img = self.transform_color(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class ImageFolderTruncated(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, dataidxs=None, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(ImageFolderTruncated, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.dataidxs = dataidxs

        ### we need to fetch training labels out here:
        self._train_labels = np.array([tup[-1] for tup in self.imgs])

        self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.dataidxs is not None:
            #self.imgs = self.imgs[self.dataidxs]
            self.imgs = [self.imgs[idx] for idx in self.dataidxs]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    @property
    def get_train_labels(self):
        return self._train_labels