from qgis.core import QgsApplication
from qgis.gui import QgisInterface
from qgis.PyQt.QtCore import pyqtSignal, QObject
from qgis.PyQt.QtWidgets import (
    QAction,
    QToolBar,
)
import processing

from .tools.widgetTool import Selector
from .ui.icons import QIcon_GeoSAMTool, QIcon_EncoderTool
from .geo_sam2_provider import GeoSam2Provider
from .tools.messageTool import MessageTool


class Geo_SAM2(QObject):
    execute_SAM = pyqtSignal()

    def __init__(self, iface: QgisInterface, cwd: str):
        super().__init__()
        self.iface = iface
        self.cwd = cwd
        self.canvas = iface.mapCanvas()

        # 初始化为 None，防止未调用 initGui 时报错
        self.actionSamTool = None
        self.actionSamEncoder = None
        self.toolbar = None
        self.provider = None

    def initProcessing(self):
        self.provider = GeoSam2Provider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()

        self.toolbar: QToolBar = self.iface.addToolBar('Geo-SAM2 Toolbar')
        self.toolbar.setObjectName('mGeoSamToolbar')
        self.toolbar.setToolTip('Geo-SAM2 Toolbar')

        self.actionSamTool = QAction(
            QIcon_GeoSAMTool,
            "Geo-SAM2 Segmentation",
            self.iface.mainWindow()
        )
        self.actionSamEncoder = QAction(
            QIcon_EncoderTool,
            "Geo-SAM2 Image Encoder",
            self.iface.mainWindow()
        )

        self.actionSamTool.setObjectName("mActionGeoSamTool")
        self.actionSamTool.setToolTip(
            "Geo-SAM2 Segmentation: Use it to label landforms")
        self.actionSamTool.triggered.connect(self.create_widget_selector)

        self.actionSamEncoder.setObjectName("mActionGeoSamEncoder")
        self.actionSamEncoder.setToolTip(
            "Geo-SAM2 Image Encoder: Use it to encode/preprocess image before labeling")
        self.actionSamEncoder.triggered.connect(self.encodeImage)

        self.iface.addPluginToMenu('Geo-SAM2 Tools', self.actionSamTool)
        self.iface.addPluginToMenu('Geo-SAM2 Tools', self.actionSamEncoder)

        # self.iface.addToolBarIcon(self.action)
        self.toolbar.addAction(self.actionSamTool)
        self.toolbar.addAction(self.actionSamEncoder)
        self.toolbar.setVisible(True)

    def create_widget_selector(self):
        '''Create widget for selecting landform by prompts'''
        if not hasattr(self, "wdg_select"):
            self.wdg_select = Selector(self, self.iface, self.cwd)
        self.wdg_select.open_widget()

    def unload(self):
        '''Unload actions when plugin is closed'''
        if hasattr(self, "wdg_select"):
            self.wdg_select.unload()
            self.wdg_select.setParent(None)

        # 安全判断属性是否存在且不为None
        if getattr(self, "actionSamTool", None):
            self.iface.removeToolBarIcon(self.actionSamTool)
            self.iface.removePluginMenu('&Geo-SAM2 Tools', self.actionSamTool)
            del self.actionSamTool
        if getattr(self, "actionSamEncoder", None):
            self.iface.removeToolBarIcon(self.actionSamEncoder)
            self.iface.removePluginMenu('&Geo-SAM2 Tools', self.actionSamEncoder)
            del self.actionSamEncoder
        if getattr(self, "toolbar", None):
            del self.toolbar
        if getattr(self, "provider", None):
            QgsApplication.processingRegistry().removeProvider(self.provider)
            del self.provider

    def encodeImage(self):
        '''Convert layer containing a point x & y coordinate to a new point layer'''
        processing.execAlgorithmDialog('geo_sam2:geo_sam2_encoder', {})
