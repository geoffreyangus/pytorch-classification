from emmental.contrib.slicing.slicing_function import slicing_function


# This is an example slicing function, free feel to add more...
@slicing_function(fields=["image_name"])
def slice_example(example):
    return "1" in example.image_name


# Put all slicing functions you want to include in this dictionary
slicing_function_dict = {"Atelectasis": {"slice_example": slice_example}}
