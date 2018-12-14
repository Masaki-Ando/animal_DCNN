import os
import sys
import argparse
import keras
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from data_generator import DataGenerator


SUBSTANCES = ['unclear_a', 'mustelidae', 'boar', 'bird', 'deer', 'masked',
              'fox', 'raccoondog', 'serow', 'human', 'rabbit',
              'squirrel', 'bear', 'mouse', 'monkey', 'bat', 'dog', 'cat'
              ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_json', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--weight_path', type=str)
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--normalization', type=str, default='sigmoid')
    parser.add_argument('--substances', '-s', nargs='+', default=SUBSTANCES)
    parser.add_argument('--csv_path', default='./csv/DL_base_171228_lp.csv')
    parser.add_argument('--batch_size', '-bs', type=int, default=16)
    parser.add_argument('--height', '-ht', type=int, default=224)
    parser.add_argument('--width', '-wd', type=int, default=224)

    args = parser.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    # import model from json file
    model = keras.models.model_from_json(open(args.model_json).read())
    # load parameters of model
    model.load_weights(args.weight_path)

    target_size = (args.width, args.height)

    # lead csv file
    df = pd.read_csv(args.csv_path)

    # Prediction of each of train, validation and test dataset is make.
    # Each result is exported to csv file.
    # Threshold is determined by the result of validation.
    for i, phase in enumerate(['train', 'valid', 'test']):
        # create generator for predicting
        data_generator = DataGenerator(args.substances, target_size, phase) \
            .flow_from_csv(args.csv_path, args.batch_size,
                           normalization=args.normalization,
                           task=args.task,
                           is_shuffle=False)

        print('Phase: {}'.format(phase))
        # predict
        predicted = model.predict_generator(data_generator,
                                            data_generator.steps_per_epoch,
                                            verbose=1)

        # extract rows
        _df = pd.DataFrame(df.loc[df['learning_phase'] == i])[['fullpath'] + args.substances]

        # export result
        df_predicted = pd.DataFrame(_df['fullpath'])
        for si, substance in enumerate(args.substances):
            kwargs = {substance: predicted[si]}
            df_predicted = df_predicted.assign(**kwargs)
        df_predicted.to_csv(os.path.join(args.result_dir, 'predicted_{}.csv'.format(phase)),
                            index_label='original_index')

        # export result (discrete)
        for substance in args.substances:
            df_predicted[substance] = df_predicted[substance].map(lambda x: np.abs(round(x)))
        df_predicted.to_csv(os.path.join(args.result_dir, 'predicted_{}_discrete.csv'.format(phase)),
                            index_label='original_index')


if __name__ == '__main__':
    main()
