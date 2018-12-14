# animal_DCNN
This program was used in the paper below, and is published according to MIT license.  

`
Ando M., Nakatsuka S., Aizawa H., Nakamori S., Ikeda T., Moribe J., Terada K. and Kato K. In_submitting. Recognition of wildlife in images taken by camera trap with deep learning.
`

このプログラムは以下の論文で使用されたものであり、MITライセンスにしたがって公開されています。  
`
安藤正規・中塚俊介・相澤宏旭・中森さつき・池田敬・森部絢嗣・寺田和憲・加藤邦人. 投稿中. 深層学習（Deep Learning）によるカメラトラップ画像の判別.哺乳類科学
`



## Hardware Environments 物理環境
* CPU: Intel i5-7600
* Memory: 32GB
* GPU: nvidia GeForce GTX 1080

## Software Environments ソフトウェア環境
* Ubuntu 16.04LTS(64bit)
* GPU Driver nvidia-396
* python 3.5
* CUDA 9.0
* cudnn v7.3
* tensorflow 1.10
* keras 2.2.4

## Data format データ形式
See sample.csv  
sample.csvを参照してください。

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

## How to use 使用方法
### training 学習
```
usage: train.py [-h] [--substances SUBSTANCES [SUBSTANCES ...]]
                [--nb_epoch NB_EPOCH] [--batch_size BATCH_SIZE]
                [--height HEIGHT] [--width WIDTH] [--save_steps SAVE_STEPS]
                [--csv_path CSV_PATH] [--param_dir PARAM_DIR]
                [--optimizer {sgd,adam,adadelta}]
                [--learning_rate LEARNING_RATE] [--weight WEIGHT]
                [--freeze_index FREEZE_INDEX] [--base BASE] [--task TASK]

optional arguments:
  -h, --help            show this help message and exit
  --substances SUBSTANCES [SUBSTANCES ...], -s SUBSTANCES [SUBSTANCES ...]
                        回帰する動物種（1種以上の指定が必要）
  --nb_epoch NB_EPOCH, -e NB_EPOCH
                        学習回数
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        バッチサイズ
  --height HEIGHT, -ht HEIGHT
                        ネットワークに入力する画像の高さ（内部でこの高さにリサイズ）
  --width WIDTH, -wd WIDTH
                        ネットワークに入力する画像の幅（内部でこの幅にリサイズ）
  --save_steps SAVE_STEPS, -ss SAVE_STEPS
                        何Epochごとに重みを保存するか
  --csv_path CSV_PATH   学習の際に使用するCSVファイルのパス
  --param_dir PARAM_DIR
                        重みを保存するディレクトリ
  --optimizer {sgd,adam,adadelta}
                        最適化手法の指定．以下の3種から指定 [sgd, adam, adadelta]
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        学習率の指定．デフォルトはkerasのデフォルト値に従う
  --weight WEIGHT       初期重みの設定．kerasの学習済みモデルから学習する場合は imagenet を指定．
  --freeze_index FREEZE_INDEX, -fi FREEZE_INDEX
                        ResNetやVGGをボトムから数えて何ブロック目まで，Fixするか
  --base BASE           ベースネットワークの指定．以下の2種から指定 [resnet50, vgg16]
  --task TASK           タスクの指定．本論文では multi_regression を使用

```

### prediction 予測
```
usage: predict.py [-h] [--model_json MODEL_JSON] [--task TASK]
                  [--weight_path WEIGHT_PATH] [--result_dir RESULT_DIR]
                  [--normalization NORMALIZATION]
                  [--substances SUBSTANCES [SUBSTANCES ...]]
                  [--csv_path CSV_PATH] [--batch_size BATCH_SIZE]
                  [--height HEIGHT] [--width WIDTH]

optional arguments:
  -h, --help            show this help message and exit
  --model_json MODEL_JSON
                        train.pyで保存されたモデルのJSONファイルのパス
  --task TASK           タスクの指定．本論文では multi_regression を使用
  --weight_path WEIGHT_PATH
                        train.pyで保存されたモデルの重みファイルのパス
  --result_dir RESULT_DIR
                        予測結果のCSVファイルを保存するディレクトリ
  --normalization NORMALIZATION
                        入力画像の正規化のスケール設定．学習時に imagenet の重みを初期値にしたなら，imagenet
                        を指定．[0, 1]なら sigmoid を指定
  --substances SUBSTANCES [SUBSTANCES ...], -s SUBSTANCES [SUBSTANCES ...]
                        回帰する動物種（1種以上の指定が必要）
  --csv_path CSV_PATH   学習の際に使用するCSVファイルのパス
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        バッチサイズ
  --height HEIGHT, -ht HEIGHT
                        ネットワークに入力する画像の高さ（内部でこの高さにリサイズ）
  --width WIDTH, -wd WIDTH
                        ネットワークに入力する画像の幅（内部でこの幅にリサイズ）

```

## Model モデル
Base network is resnet50.  
This the paper, we used multi-regression model.  
This network outputs a N-dim vector from an image, each element of that vector represents the number of each animal in the input image.  
このモデルはresnet50のネットワーク構造をベースとしている。  
論文中では、多項回帰モデルを設定してモデルを学習した。  
モデルの出力は、画像内に存在すると予測される各動物種の頭数のn次元ベクトルである。  

e.g.) regression of the numbers of ['deer', 'serow', 'boar', 'bear']  
```
output = (0.023, 0.082, 1.213, 0.001)  
threshold = 0.1  
```

* output[0] shows the number of deer (=0).
* output[1] shows the number of serow (=0).
* output[2] shows the number of boar (=1).
* output[3] shows the number of bear (=0).

例) [シカ,カモシカ,イノシシ,ツキノワグマ]の頭数を回帰した場合、出力(output)、閾値(threshold)が上記のとおりだった場合、  

* output[0] はシカの頭数 (=0)
* output[1] はカモシカの頭数 (=0)
* output[2] はイノシシの頭数 (=1)
* output[3] はツキノワグマの頭数 (=0)  
の推定値をそれぞれ示す。

