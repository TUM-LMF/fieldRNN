#!/bin/sh

# download neural network models
wget ftp://ftp.lrz.de/transfer/fieldRNN/models.zip
unzip models.zip
mv folds models
rm models.zip

# download svm model
mkdir svm
cd svm
wget ftp://ftp.lrz.de/transfer/fieldRNN/svm.zip
unzip svm.zip
rm svm.zip
