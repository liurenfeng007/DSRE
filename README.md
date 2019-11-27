#  Distant Supervision for Relation Extraction

## Introduction
A framework for Distant Supervision for Relation Extraction

## Overview
It is a framework for easily build relation extraction model.

* data
  * 
* models
  * base_model
  * pcnn_one
  * pcnn_att
* module
  * classifier
  * embedding
  * encoder
  * pooling
  * selector
* raw_data
  * 

## Requirements
- python 3.6
- pytorch 0.31

## Dataset
OpenNRE original version dataset

## Usage
###Generate processed data
```bash
python gen_data.py
```
### Train Model
```
python train.py 
```
### Test Model
```bash
python test.py 
```
### NYT10 Dataset



## Citation
