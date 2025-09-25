# -----------------------------------------------------------
# Copyright (C) 2023 CryoLab CUHK
# -----------------------------------------------------------
import os
import inspect
from .geo_sam2_tool import Geo_SAM2

cmd_folder = os.path.split(inspect.getfile(inspect.currentframe()))[0]


def classFactory(iface):
    return Geo_SAM2(iface, cmd_folder)
