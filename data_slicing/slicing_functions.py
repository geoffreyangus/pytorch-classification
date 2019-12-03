from collections import defaultdict
from functools import partial

import pandas as pd

from emmental.contrib.slicing.slicing_function import slicing_function

# Creating global df from nih labels
# TODO: Replace this with external data structure, for now load from local
SLICE_DF = None


@slicing_function(fields=["image_name"])
def slice_drain(example):
    return SLICE_DF.loc[example.image_name]['drain'] == 1


@slicing_function(fields=["image_name"])
def slice_atelectasis(example):
    return SLICE_DF.loc[example.image_name]["Atelectasis"] == 1


@slicing_function(fields=["image_name"])
def slice_cardiomegaly(example):
    return SLICE_DF.loc[example.image_name]["Cardiomegaly"] == 1


@slicing_function(fields=["image_name"])
def slice_effusion(example):
    return SLICE_DF.loc[example.image_name]["Effusion"] == 1


@slicing_function(fields=["image_name"])
def slice_infiltration(example):
    return SLICE_DF.loc[example.image_name]["Infiltration"] == 1


@slicing_function(fields=["image_name"])
def slice_mass(example):
    return SLICE_DF.loc[example.image_name]["Mass"] == 1


@slicing_function(fields=["image_name"])
def slice_nodule(example):
    return SLICE_DF.loc[example.image_name]["Nodule"] == 1


@slicing_function(fields=["image_name"])
def slice_pneumonia(example):
    return SLICE_DF.loc[example.image_name]["Pneumonia"] == 1


@slicing_function(fields=["image_name"])
def slice_pneumothorax(example):
    return SLICE_DF.loc[example.image_name]["Pneumothorax"] == 1


@slicing_function(fields=["image_name"])
def slice_consolidation(example):
    return SLICE_DF.loc[example.image_name]["Consolidation"] == 1


@slicing_function(fields=["image_name"])
def slice_edema(example):
    return SLICE_DF.loc[example.image_name]["Edema"] == 1


@slicing_function(fields=["image_name"])
def slice_emphysema(example):
    return SLICE_DF.loc[example.image_name]["Emphysema"] == 1


@slicing_function(fields=["image_name"])
def slice_fibrosis(example):
    return SLICE_DF.loc[example.image_name]["Fibrosis"] == 1


@slicing_function(fields=["image_name"])
def slice_pleural_thickening(example):
    return SLICE_DF.loc[example.image_name]["Pleural_Thickening"] == 1


@slicing_function(fields=["image_name"])
def slice_hernia(example):
    return SLICE_DF.loc[example.image_name]["Hernia"] == 1


@slicing_function(fields=["image_name"])
def slice_normal(example):
    return SLICE_DF.loc[example.image_name]["Abnormal"] == 0


@slicing_function(fields=["image_name"])
def slice_concerning_effusion(example):
    effusion = SLICE_DF.loc[example.image_name]["Effusion"] == 1
    edema = SLICE_DF.loc[example.image_name]["Edema"] == 1
    pneumonia = SLICE_DF.loc[example.image_name]["Pneumonia"] == 1
    concerning_effusion = effusion and (pneumonia or edema)
    return concerning_effusion


# @slicing_function(fields=["image_name"])
# def slice_pneumothorax(example):
#    return SLICE_DF[example.image_name]['Atelectasis'] == 1
