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
    parser.add_argument('--model_json', type=str, help='train.pyで保存されたモデルのJSONファイルのパス')
    parser.add_argument('--task', default='multi_regression', help='タスクの指定．本論文では multi_regression を使用')
    parser.add_argument('--weight_path', type=str, help='train.pyで保存されたモデルの重みファイルのパス')
    parser.add_argument('--result_dir', type=str, help='予測結果のCSVファイルを保存するディレクトリ')
    parser.add_argument('--normalization', type=str, default='imagenet', help='入力画像の正規化のスケール設定．学習時に imagenet の重みを初期値にしたなら，imagenet を指定．[0, 1]なら sigmoid を指定')
    parser.add_argument('--substances', '-s', nargs='+', default=SUBSTANCES, help='回帰する動物種（1種以上の指定が必要）')
    parser.add_argument('--csv_path', default='./csv/DL_base_171228_lp.csv', help='学習の際に使用するCSVファイルのパス')
    parser.add_argument('--batch_size', '-bs', type=int, default=16, help='バッチサイズ')
    parser.add_argument('--height', '-ht', type=int, default=224, help='ネットワークに入力する画像の高さ（内部でこの高さにリサイズ）')
    parser.add_argument('--width', '-wd', type=int, default=224, help='ネットワークに入力する画像の幅（内部でこの幅にリサイズ）')
    
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
