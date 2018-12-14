from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Input, Dense, Flatten


def get_model(input_shape,
              nb_classes,
              base='vgg16',
              weights=None,
              task='multi_regression',
              freeze_index=None,
              is_plot=True,
              labels=None):
    """
    Construct keras.Model.

    Args:
        input_shape: shape of input (h, w, c)
        nb_classes: number of classes
        base: base model. You can use 'vgg16' or 'resnet50'
        weights: path of parameter file of base model. You can use filepath or 'imagenet'
        task: You can use
                'classification',
                'multi_regression',
                total_regression',
                'present_absent'
        freeze_index: Index when the model is divided into 5 for each resolution.
                      The weights are fixed before this index.
        is_plot: whether you plot model
        labels: labels of output layers

    Returns: keras.Model

    """
    # prepare labels of output layer
    labels = [None] * nb_classes if labels is None else labels
    assert len(labels) == nb_classes

    # only display message
    if weights is not None:
        print('LOADING {} ...'.format(weights))

    # VGG16
    if base == 'vgg16':
        base_model = VGG16
        # get the index
        # freeze the layer before the index
        if freeze_index is not None:
            fl_index = [3, 6, 10, 14, 18][freeze_index]
        else:
            fl_index = None

    # ResNet50
    elif base == 'resnet50':
        base_model = ResNet50
        # get the index
        # freeze the layer before the index
        if freeze_index is not None:
            fl_index = [5, 36, 78, 140, 172][freeze_index]
        else:
            fl_index = None
    else:
        raise NotImplementedError

    # input layer
    input_ = Input(input_shape)
    # build base model
    x = base_model(include_top=False, weights=weights, input_tensor=input_).output

    # flatten
    # (bs, h, w, c) => (bs, -1)
    x = Flatten()(x)
    # full-connect
    x = Dense(1024, activation='relu')(x)

    # Output shape is depended on the task.
    # N-class classification
    # output: (bs, nb_classes)
    if task == 'classification':
        outputs = Dense(nb_classes, activation='softmax')(x)

    # Regression of count of each substance
    # output: List containing the outputs of nb_classes layers
    elif task == 'multi_regression':
        outputs = [Dense(1, activation='linear', name=label)(x)
                   for label in labels]

    # Regression of count of total
    # output: (bs, )
    elif task == 'total_regression':
        outputs = Dense(1, activation='linear')(x)

    # Binary classification (present / absent)
    # output: (bs, )
    elif task == 'present_absent':
        outputs = Dense(1, activation='sigmoid')(x)

    else:
        raise NotImplementedError

    # build whole model
    model = Model(inputs=input_, outputs=outputs)

    # plot model by pydot
    if is_plot:
        try:
            from utils.custom_vis import plot_model
            plot_model(model, to_file='model.svg')
            plot_model(model, to_file='model.png')
        except:
            pass

    # freeze layers
    if fl_index is not None:
        for i in range(fl_index+1):
            model.layers[i].trainable = False

    return model


if __name__ == '__main__':
    get_model((224, 224, 3), 2)
