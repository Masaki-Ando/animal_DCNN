import pandas as pd
import codecs as cd
import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import Iterator
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import np_utils
import copy


ROOT_DIR = os.environ['HOME']
# ROOT_DIR = os.path.join(os.environ['DS'], 'animal')
SUBSTANCES = ['unclear_a', 'mustelidae', 'boar', 'bird', 'deer', 'masked',
              'fox', 'raccoondog', 'serow', 'human', 'rabbit',
              'squirrel', 'bear', 'mouse', 'monkey', 'bat', 'dog', 'cat'
              ]


class DataGenerator:
    def __init__(self, substance_list,
                 target_size,
                 learning_phase='train'):
        """Class for keras.Model.fit_generator

        Args:
            substance_list (list): list of target animals
            target_size (tuple): size of input image after resizing
            learning_phase (str): one of ['train', 'validation', 'test']
        """
        assert learning_phase in ['train', 'valid', 'test']
        self.substance_list = substance_list
        self.target_size = target_size
        self.data_num = None
        self.learning_phase = learning_phase

    def flow_from_csv(self, csv_path,
                      batch_size,
                      is_shuffle=True,
                      normalization='sigmoid',
                      task='multi_regression'):
        """Method for getting DataIterator

        Args:
            csv_path: csv path
            batch_size: batch size
            is_shuffle: whether you'd like to shuffle indices of data
            normalization: normalization mode.
                           You can use 'sigmoid' or 'imagenet'.
                           - sigmoid: 0. - 1.
                           - imagenet: centering images using parameters of imagenet
            task: 'classification', 'multi-regression',
                  'total_regression', 'present_absent'

        Returns:
            DataIterator
        """
        df = read_csv(csv_path)
        paths, labels = self._parse_csv(df, task)
        nb_classes = len(self.substance_list)
        return DataIterator(paths, labels, nb_classes,
                            target_size=self.target_size,
                            batch_size=batch_size, shuffle=is_shuffle,
                            normalization=normalization,
                            task=task)

    def _parse_csv(self, df,
                   task):
        """Method feeding paths and labels corresponding to task

        Args:
            df (pandas.DataFrame)
            task (str)

        Returns:
            paths (numpy.ndarray)
            labels (numpy.ndarray)
        """
        mode_index = ['train', 'valid', 'test'].index(self.learning_phase)

        # N-class classification
        if task == 'classification':
            # extract rows with the specified class
            _df = df[df['substance'].isin(self.substance_list)]
            # extract rows with the specified learning_phase
            _df = _df[_df['learning_phase'] == mode_index]
            # concat
            _df = _df[['substance', 'fullpath']]

            # replace substance(str) to
            for label, sub in enumerate(self.substance_list):
                _df.loc[(_df['substance'] == sub), 'substance'] = label

            # to ndarray
            paths = np.array(_df['fullpath'])
            labels = np.array(_df['substance'])
            return paths, labels

        # Regression of count of each substance
        elif task == 'multi_regression':
            # extract rows with the specified learning_phase
            _df = df.loc[df['learning_phase'] == mode_index]
            # extract cols
            _df = _df[self.substance_list + ['fullpath']]

            # to ndarray
            paths = np.array(_df['fullpath'])
            labels = np.array(_df[self.substance_list])
            return paths, labels

        # Regression of count of total
        elif task == 'total_regression':
            # extract rows with the specified learning_phase
            _df = df.loc[df['learning_phase'] == mode_index]
            # extract cols
            _df = _df[self.substance_list + ['fullpath']]
            # compute the sum of substances and concat
            _df = pd.concat([_df['fullpath'],
                             pd.DataFrame(_df[self.substance_list].sum(axis=1), columns=['total'])],
                            axis=1)

            # to ndarray
            paths = np.array(_df['fullpath'])
            labels = np.array(_df['total'])
            return paths, labels

        # Binary classification (present / absent)
        elif task == 'present_absent':
            # extract rows with the specified learning_phase
            _df = df.loc[df['learning_phase'] == mode_index]
            # extract cols
            _df = _df[self.substance_list + ['fullpath']]
            # compute "present or absent" and concat
            _df = pd.concat([_df['fullpath'],
                             pd.DataFrame(_df[self.substance_list].sum(axis=1), columns=['total'])],
                            axis=1)

            # to ndarray
            paths = np.array(_df['fullpath'])
            labels = np.array(_df['total'], dtype='bool').astype('float32')
            return paths, labels
        else:
            raise NotImplementedError


class DataIterator(Iterator):
    def __init__(self, paths, labels, nb_classes, target_size, batch_size, shuffle,
                 normalization='sigmoid', task='multi_regression', seed=None):
        self.paths = paths
        self.labels = labels
        self.target_size = target_size
        self.nb_classes = nb_classes
        self.task = task
        self.normalization = normalization
        self._current_paths = None

        # calculate the number of iterations per an epoch
        self.steps_per_epoch = len(paths) // batch_size
        if len(paths) % batch_size != 0:
            self.steps_per_epoch += 1

        super().__init__(len(self.paths), batch_size, shuffle, seed)

    def next(self):
        """Method for iteration.

        Returns: (x, y)
        """

        # get indices
        # (self.batch_size, )
        with self.lock:
            # index_array, _, _ = next(self.index_generator)   
            index_array = next(self.index_generator)   

        # get paths of images on an iteration
        image_path_batch = self.paths[index_array]
        self._current_paths = image_path_batch

        # create batch
        # image: (bs, h, w, c)
        # label: depends on the task
        image_batch = np.array([self.load_image(path, self.normalization)
                                for path in image_path_batch])
        label_batch = copy.deepcopy(self.labels[index_array])

        if self.task == 'multi_regression':
            # (bs, nb_substances) => (nb_substances, bs)
            # This is because model has nb_substances output layers.
            label_batch = list(label_batch.transpose())

        elif self.task == 'classification':
            # (bs, ) => (bs, nb_substances)
            # one-hot vectorize
            label_batch = np_utils.to_categorical(label_batch, self.nb_classes)

        elif self.task == 'total_regression':
            pass

        elif self.task == 'present_absent':
            pass

        else:
            raise NotImplementedError

        return image_batch, label_batch

    def load_image(self, path, mode='sigmoid'):
        """Method for loading image from path and normalize it.

        Args:
            path: image path
            mode: normalization mode. 'sigmoid' or 'imagenet'

        Returns: normalized image

        """
        # replace "~" to ROOT_DIR
        _path = path.replace('~', ROOT_DIR)
        # load image
        image = Image.open(_path)
        # resize
        image = image.resize(self.target_size, resample=Image.BILINEAR)
        # to ndarray
        array = np.asarray(image)

        # [0, 255] => [0., 1.]
        if mode == 'sigmoid':
            array = array.astype('float32') / 255

        # normalize by the mean of imagenet
        # mean = [103.939, 116.779, 123.68]
        # Zero-center by mean pixel
        elif mode == 'imagenet':
            array = array.astype('float32')
            array = np.expand_dims(array, axis=0)
            array = preprocess_input(array)
            array = np.squeeze(array, axis=0)
        else:
            raise NotImplementedError
        return array

    @property
    def current_paths(self):
        return self._current_paths


def read_csv(csv_path):
    with cd.open(csv_path, "r", "Shift-JIS", "ignore") as csv_file:
        df = pd.read_csv(csv_file)
    return df


def main():
    csv_path = './csv/DL_base_171228_lp.csv'
    dg = DataGenerator(SUBSTANCES, (224, 224), 'valid')
    di = dg.flow_from_csv(csv_path, 64, task='total_regression')

    while True:
        i, l = next(di)
        print(l)


if __name__ == '__main__':
    main()
