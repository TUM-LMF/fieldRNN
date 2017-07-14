# Recurrent Neural Networks for Multitemporal Crop Identification

##### Source code of Rußwurm & Körner (2017) at [EARTHVISION 2017](https://www.grss-ieee.org/earthvision2017/)

When you use this code please cite
```
Rußwurm M., Körner M. (2017). Temporal Vegetation Modelling using Long Short-Term Memory Networks
for Crop Identification from Medium-Resolution Multi-Spectral Satellite Images. In Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2017.
```

#### Tensorflow Graphs
The TensorFlow graphs for recurrent and convolutional networks are defined at ```rnn_model.py``` and ```cnn_model.py```.

#### Installation

**Please Note**: The data of this project is stored in local PostGIS Database which
the provided scripts are accessing.
Hence currently the ```train``` and ```evaluation``` scripts are not executable, due to database restrictions.
However the training and evalutation will be published shortly.

##### Requirements

* [Tensorflow == 1.0.1](https://www.tensorflow.org/)
* [scikit-learn >= 0.18.1](http://scikit-learn.org/stable/)
* [numpy >= 1.11.2](http://www.numpy.org/)
* [pandas >= 0.19.1](http://pandas.pydata.org/)

A complete package list at ```requirements.txt```

<div class="alert alert-warning">
Due to changes in the tf.nn.rnn_cell.MultiRNN class in Tensorflow 1.2.0 the current code is not compatible with TF version 1.2.0
</div>

##### Clone this repository
```
git clone https://github.com/TUM-LMF/fieldRNN.git
```

<!--
download and unzip training data to ```data/```
```
wget LoremIpsum
```
-->

#### Network Training
The training is performed on *train* data, either from the database directly. The *test* (also referred to as *validation*) data is used logged in Tensorflow event files.

```
$ python train.py --help
> positional arguments:
> layers                number of layers
> cells                 number of rnn cells, as multiple of 55
> dropout               dropout keep probability
> fold                  select training/evaluation fold to use
> maxepoch              maximum epochs
> batchsize             batchsize
```

For instance:
```
python train.py 4 2 0.5 0 30 500 --gpu 1 --model lstm --savedir save --datadir data;
```
tensorflow checkpoint and eventfiles of this call will be stored at ```save/lstm/4l2r50d0f```

### Model Evaluation
The script ```evaluate.py``` evaluates one model based on *evaluation* data.

```
python evaluate.py save/folds/lstm/2l4r50d9f
```
The latest checkpoint of one model is restored and the entire body of *evaluation* data is processed.
After the evaluation process ```eval_targets.npy```, ```eval_probabilities.npy``` and ```eval_observations.npy``` are stores in the ```save``` directory.
These files are later used for calculation of accuracy metrics by ```cvprwsevaluation.ipynb```

## Support Vector Machine baseline
Support Vector Machine for baseline evaluation is based on [scikit-learn](http://scikit-learn.org/stable/) framework

The script ```svm.py``` performes the gridsearch.
The generated files ```svm/scores.npy```, ```svm/targets.npy```,  ```svm/predicted.npy``` are needed for ```cvprwsevaluation.ipynb```

## Earthvision 2017 Evaluation

The (complete, but untidy) evaluation of plots and accuracy metrics can be founds at ```cvprwsevaluation.ipynb```
