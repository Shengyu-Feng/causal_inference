# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class DataManager(object):
  def load(self):
    # Load dataset
    dataset = np.load('../states.npy')

    # print('Keys in the dataset:', dataset_zip.keys())
    #  ['metadata', 'imgs', 'latents_classes', 'latents_values']

    self.imgs = dataset.astype(float)/255

    self.n_samples = dataset.shape[0]
    # 1e6
    
  @property
  def sample_size(self):
    return self.n_samples

  def get_image(self):
    return self.get_images([0])[0]

  def get_images(self, indices):
    images = []
    for index in indices:
      img = self.imgs[index]
      img = img.reshape(6400)
      images.append(img)
    return images

  def get_random_images(self, size):
    indices = [np.random.randint(self.n_samples) for i in range(size)]
    return self.get_images(indices)
