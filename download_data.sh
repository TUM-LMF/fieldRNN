#!/bin/sh

mkdir data
cd data
wget ftp://ftp.lrz.de/transfer/fieldRNN/dataset.zip
unzip dataset.zip
rm dataset.zip
