# Multi-Label-Classification Without Deep Learning

This repo contains some rough experiments on a relatively small multi-label classification dataset. The goal is to show some of the difficulties that arise
with multi-label classification, and how these difficulties can be overcome using relatively unknown techiques that don't require transfer learning using
large language models. The code is in no way production quality and shouldn't be interpreted as such. For anyone reading, you may find the commentary in notebooks
informative but the code is not.

## Structure
Since this repo is mostly a collection of experiments for personal use, it's mostly unstructured. The training pipeline can be found in src/sklearn_trainer.py
and the results

## General setup considerations.

### Installing XGBoost from source

* [Here](https://xgboost.readthedocs.io/en/stable/build.html) is a quick guide to installing XGBoost from source for leveraging GPU.
* Note, if you're using a linux machine for xgboost you may need an earlier gcc compiler. These are helpful links [here](https://askubuntu.com/questions/1039856/downgrade-gnu-compilers-ubuntu-18-04) and [here](https://linuxconfig.org/how-to-switch-between-multiple-gcc-and-g-compiler-versions-on-ubuntu-20-04-lts-focal-fossa) In general, you don't want to downgrade for the system,
so follow the links for switching.



