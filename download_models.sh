#!/bin/sh

# download neural network models
wget ftp://m1370728:m1370728@138.246.224.34/models.zip
unzip models.zip
rm models.zip

wget ftp://m1370728:m1370728@138.246.224.34/models.sha512
sha512sum -c models.sha512


# download svm model
wget ftp://m1370728:m1370728@138.246.224.34/svm.zip
unzip svm.zip
rm svm.zip

wget ftp://m1370728:m1370728@138.246.224.34/svm.sha512
sha512sum -c svm.sha512
