import argparse
import os
from keras.callbacks import CSVLogger
from keras.optimizers import Adam, SGD, Adadelta
from data_generator import DataGenerator
from model import get_model
from callbacks import ModelSaver, BatchLogger
from utils.config import dump_config


SUBSTANCES = ['unclear_a', 'mustelidae', 'boar', 'bird', 'deer', 'masked',
              'fox', 'raccoondog', 'serow', 'human', 'rabbit',
              'squirrel', 'bear', 'mouse', 'monkey', 'bat', 'dog', 'cat'
              ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--substances', '-s', nargs='+', default=SUBSTANCES)
    parser.add_argument('--nb_epoch', '-e', type=int, default=50)
    parser.add_argument('--batch_size', '-bs', type=int, default=16)
    parser.add_argument('--height', '-ht', type=int, default=224)
    parser.add_argument('--width', '-wd', type=int, default=224)
    parser.add_argument('--save_steps', '-ss', type=int, default=10)
    parser.add_argument('--csv_path', default='./csv/DL_base_171228_lp.csv')
    parser.add_argument('--param_dir', default='./output/params')
    parser.add_argument('--optimizer', default='adadelta', choices=['sgd', 'adam', 'adadelta'])
    parser.add_argument('--learning_rate', '-lr', type=float, default=None)
    parser.add_argument('--weight', default=None)
    parser.add_argument('--freeze_index', '-fi', type=int, default=None)
    parser.add_argument('--base', default='vgg16')
    parser.add_argument('--task', default='multi_regression')

    args = parser.parse_args()
    target_size = (args.width, args.height)
    input_shape = (args.height, args.width, 3)

    # get model
    model = get_model(input_shape,
                      len(args.substances),
                      base=args.base,
                      weights=args.weight,
                      freeze_index=args.freeze_index,
                      task=args.task,
                      labels=args.substances)

    # make directory saving outputs
    os.makedirs(args.param_dir, exist_ok=True)

    # export model to json file
    open(os.path.join(args.param_dir, 'model.json'), 'w').write(model.to_json())

    # save training config
    dump_config(os.path.join(args.param_dir, 'config.csv'), args)

    # select mode of normalization
    normalization = 'imagenet' if args.weight == 'imagenet' else 'sigmoid'

    # create generator for training
    train_data_generator = DataGenerator(args.substances, target_size, 'train')\
        .flow_from_csv(args.csv_path, args.batch_size,
                       normalization=normalization,
                       task=args.task)
    # create generator for validation
    valid_data_generator = DataGenerator(args.substances, target_size, 'valid')\
        .flow_from_csv(args.csv_path, args.batch_size,
                       normalization=normalization,
                       task=args.task)

    # create callbacks
    callbacks = [CSVLogger(os.path.join(args.param_dir, "learning_log_epoch.csv")),
                 BatchLogger(os.path.join(args.param_dir, "learning_log_iter.csv")),
                 ModelSaver(os.path.join(args.param_dir, "param_{epoch:02d}.h5"),
                            save_freq=args.save_steps)]

    # create optimizer
    optimizer = get_optimizer(args.optimizer, args.learning_rate)
    # get loss and metrics
    loss, metrics = get_loss_metrics(args.task)
    # compile model using loss and metrics
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    # fit model
    model.fit_generator(train_data_generator,
                        train_data_generator.steps_per_epoch,
                        epochs=args.nb_epoch,
                        validation_data=valid_data_generator,
                        validation_steps=valid_data_generator.steps_per_epoch,
                        callbacks=callbacks)


def get_loss_metrics(task):
    """Get loss and metrics corresponding to task

    Args
        task (str): solving task

    Returns
        loss
        metrics
    """
    if task == 'multi_regression':
        loss = 'mse'
        metrics = None
    elif task == 'classification':
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    elif task == 'total_regression':
        loss = 'mse'
        metrics = None
    elif task == 'present_absent':
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        raise NotImplementedError

    return loss, metrics


def get_optimizer(optimizer, lr):
    """Get optimizer

    Args:
        optimizer (str)
        lr (float): learning rate
    Returns:
        keras.optimizer
    """
    if optimizer == 'sgd':
        opt = SGD
    elif optimizer == 'adam':
        opt = Adam
    elif optimizer == 'adadelta':
        opt = Adadelta
    else:
        raise NotImplementedError

    # If lr is not None, learning rate is set.
    if lr is not None:
        return opt(lr)
    else:
        return opt()


if __name__ == '__main__':
    main()
