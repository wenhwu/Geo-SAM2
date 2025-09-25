import os
from qgis.PyQt import uic
from qgis.gui import QgsDockWidget
# try:
#     from qgis.PyQt.QtGui import QDockWidget, QWidget
# except:
#     from qgis.PyQt.QtWidgets import QDockWidget, QWidget

cwd = os.path.abspath(os.path.dirname(__file__))
selector_path = os.path.join(cwd, "Selector_SAM2.ui")

UI_Selector: QgsDockWidget = uic.loadUi(selector_path)
