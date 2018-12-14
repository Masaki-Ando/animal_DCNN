# animal_DCNN

## Hardware Environments
* CPU: Intel i5-7600
* Memory: 32GB
* GPU: nvidia GeForce GTX 1080

## Software Environments
* Ubuntu 16.04LTS(64bit)
* GPU Driver nvidia-396
* python 3.5
* CUDA 9.0
* cudnn v7.3
* tensorflow 1.10
* keras 2.2.4

## Data format
See sample.csv

```
$ head sample.csv
fullpath,learning_phase,category,unclear_a,blank,mustelidae,boar,bird,deer,masked,fox,raccoondog,serow,human,rabbit,squirrel,bear,mouse,monkey,bat,dog,cat
~/path/to/each/photo/file/IMAG0004.JPG,0,unclear_a,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
~/path/to/each/photo/file/IMAG0005.JPG,0,unclear_a,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
~/path/to/each/photo/file/IMAG0006.JPG,0,blank,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
~/path/to/each/photo/file/IMAG0007.JPG,1,mustelidae,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
~/path/to/each/photo/file/IMAG0008.JPG,0,mustelidae,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
~/path/to/each/photo/file/IMAG0009.JPG,0,mustelidae,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
~/path/to/each/photo/file/IMAG0010.JPG,0,serow,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0
~/path/to/each/photo/file/IMAG0011.JPG,2,blank,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
~/path/to/each/photo/file/IMAG0012.JPG,0,blank,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

## How to use
### training
```
usage: train.py [-h]
                [--substances SUBSTANCES [SUBSTANCES ...]]
                [--nb_epoch NB_EPOCH]
                [--batch_size BATCH_SIZE]
                [--height HEIGHT]
                [--width WIDTH]
                [--save_steps SAVE_STEPS]
                [--csv_path CSV_PATH]
                [--param_dir PARAM_DIR]
                [--optimizer OPTIMIZER]
                [--weight WEIGHT]
                [--base BASE]
                [--vectorize VECTORIZE]
                [--task TASK]
```

### prediction
```
usage: predict.py [-h]
                  [--model_json MODEL_JSON]
                  [--task TASK]
                  [--weight_path WEIGHT_PATH]
                  [--result_dir RESULT_DIR]
                  [--normalization NORMALIZATION]
                  [--substances SUBSTANCES [SUBSTANCES ...]]
                  [--csv_path CSV_PATH]
                  [--batch_size BATCH_SIZE]
                  [--height HEIGHT]
                  [--width WIDTH]
```

## Model
This is multi-regression model.  
Base network is resnet50.  
This network outputs a N-dim vector from an image.  
Each element of that vector represents the number of each animal in the input image.  

e.g.) regression of the numbers of ['deer', 'serow', 'boar', 'bear']  
```
output = (0.023, 0.082, 1.213, 0.001)  
threshold = 0.1  
```

* output[0] shows the number of deer (=0).
* output[1] shows the number of serow (=0).
* output[2] shows the number of boar (=1).
* output[3] shows the number of cat (=0).
