# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build


def build_model(args, get_dict_batch, train_dataset):
    return build(args, get_dict_batch, train_dataset)
