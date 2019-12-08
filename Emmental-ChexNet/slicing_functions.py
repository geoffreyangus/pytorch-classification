from collections import defaultdict
from functools import partial

import pandas as pd

from emmental.contrib.slicing.slicing_function import slicing_function
from task_config import CXR8_TASK_NAMES

# Creating global df from nih labels
# TODO: Replace this with external data structure, for now load from local
DF_NIH = pd.read_csv("/lfs/1/gangus/repositories/pytorch-classification/drain_detector/data/chexnet/by-patient-id/split/all_v2.csv").set_index("Image Index")

# This is an example slicing function, free feel to add more...
@slicing_function(fields=["image_name"])
def slice_example(example):
    return "1" in example.image_name

# Defining all disease slicing functions
# @slicing_function(fields=["image_name"])
# def disease_slice(example, disease=None):
#    return df_nih[example.image_name][disease] == 1

# disease_slice_dict = {}
# for dis in CXR8_TASK_NAMES:
#    disease_slice_dict[dis.lower()] = partial(disease_slice, disease='Pneumothorax')


# Slicing function for 'drain'
@slicing_function(fields=["image_name"])
def slice_drain(example):
    return DF_NIH.loc[example.image_name]["drain"] == 1

# Slicing function for 'pneumothorax'
@slicing_function(fields=["image_name"])
def slice_atelectasis(example):
    return DF_NIH.loc[example.image_name]["Atelectasis"] == 1


@slicing_function(fields=["image_name"])
def slice_cardiomegaly(example):
    return DF_NIH.loc[example.image_name]["Cardiomegaly"] == 1


@slicing_function(fields=["image_name"])
def slice_effusion(example):
    return DF_NIH.loc[example.image_name]["Effusion"] == 1


@slicing_function(fields=["image_name"])
def slice_infiltration(example):
    return DF_NIH.loc[example.image_name]["Infiltration"] == 1


@slicing_function(fields=["image_name"])
def slice_mass(example):
    return DF_NIH.loc[example.image_name]["Mass"] == 1


@slicing_function(fields=["image_name"])
def slice_nodule(example):
    return DF_NIH.loc[example.image_name]["Nodule"] == 1


@slicing_function(fields=["image_name"])
def slice_pneumonia(example):
    return DF_NIH.loc[example.image_name]["Pneumonia"] == 1


@slicing_function(fields=["image_name"])
def slice_pneumothorax(example):
    return DF_NIH.loc[example.image_name]["Pneumothorax"] == 1


@slicing_function(fields=["image_name"])
def slice_consolidation(example):
    return DF_NIH.loc[example.image_name]["Consolidation"] == 1


@slicing_function(fields=["image_name"])
def slice_edema(example):
    return DF_NIH.loc[example.image_name]["Edema"] == 1


@slicing_function(fields=["image_name"])
def slice_emphysema(example):
    return DF_NIH.loc[example.image_name]["Emphysema"] == 1


@slicing_function(fields=["image_name"])
def slice_fibrosis(example):
    return DF_NIH.loc[example.image_name]["Fibrosis"] == 1


@slicing_function(fields=["image_name"])
def slice_pleural_thickening(example):
    return DF_NIH.loc[example.image_name]["Pleural_Thickening"] == 1


@slicing_function(fields=["image_name"])
def slice_hernia(example):
    return DF_NIH.loc[example.image_name]["Hernia"] == 1


@slicing_function(fields=["image_name"])
def slice_normal(example):
    return DF_NIH.loc[example.image_name]["Abnormal"] == 0


@slicing_function(fields=["image_name"])
def slice_concerning_effusion(example):
    effusion = DF_NIH.loc[example.image_name]["Effusion"] == 1
    edema = DF_NIH.loc[example.image_name]["Edema"] == 1
    pneumonia = DF_NIH.loc[example.image_name]["Pneumonia"] == 1
    concerning_effusion = effusion and (pneumonia or edema)
    return concerning_effusion


# Slicing function for 'pneumothorax'
# @slicing_function(fields=["image_name"])
# def slice_pneumothorax(example):
#    return df_nih[example.image_name]['Atelectasis'] == 1

abnormal_slice_dict = {
    "pneumothorax": slice_pneumothorax,
    "edema": slice_edema,
    "concerning_effusion": slice_concerning_effusion,
    "mass": slice_mass,
    "consolidation": slice_consolidation,
    "effusion": slice_effusion,
}

pneumothorax_slice_dict = {
    'drain': slice_drain
}

# Put all slicing functions you want to include for traiing in this dictionary
slicing_function_dict = defaultdict(dict)
slicing_function_dict.update({"Pneumothorax": pneumothorax_slice_dict})

# Put all slicing functions to use for evaluation in this dictionary
# abnormal_eval_slice_dict = {nm: eval(f"slice_{nm.lower()}") for nm in CXR8_TASK_NAMES}
# abnormal_eval_slice_dict.update({"concerning_effusion": slice_concerning_effusion})

# slicing_function_eval_dict = defaultdict(dict)
# slicing_function_eval_dict.update({"Abnormal": abnormal_eval_slice_dict})

drain_eval_slice_dict = {nm: eval(f"slice_{nm.lower()}") for nm in CXR8_TASK_NAMES}
drain_eval_slice_dict.update({"drain": slice_drain})

slicing_function_eval_dict = defaultdict(dict)
slicing_function_eval_dict.update({"Pneumothorax": drain_eval_slice_dict})
