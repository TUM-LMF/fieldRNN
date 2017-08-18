#!/bin/sh

wget ftp://m1370728:m1370728@138.246.224.34/data.zip
unzip data.zip
rm data.zip

# test for completeness
wget ftp://m1370728:m1370728@138.246.224.34/data.sha512
sha512sum -c data.sha512
