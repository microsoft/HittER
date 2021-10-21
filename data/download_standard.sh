#!/bin/sh

BASEDIR=`pwd`

if [ ! -d "$BASEDIR" ]; then
    mkdir "$BASEDIR"
fi


# fb15k-237
if [ ! -d "$BASEDIR/fb15k-237" ]; then
  echo Downloading fb15k-237
  cd $BASEDIR
  curl -O https://www.uni-mannheim.de/media/Einrichtungen/dws/pi1/kge_datasets/fb15k-237.tar.gz
  tar xvf fb15k-237.tar.gz
else
    echo fb15k-237 already present
fi
if [ ! -f "$BASEDIR/fb15k-237/dataset.yaml" ]; then
  python preprocess.py fb15k-237
else
    echo fb15k-237 already prepared
fi

# wnrr
if [ ! -d "$BASEDIR/wnrr" ]; then
  echo Downloading wnrr
  cd $BASEDIR
  curl -O https://www.uni-mannheim.de/media/Einrichtungen/dws/pi1/kge_datasets/wnrr.tar.gz
  tar xvf wnrr.tar.gz
else
    echo wnrr already present
fi
if [ ! -f "$BASEDIR/wnrr/dataset.yaml" ]; then
  python preprocess.py wnrr
else
    echo wnrr already prepared
fi