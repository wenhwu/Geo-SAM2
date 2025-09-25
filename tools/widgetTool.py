import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import rasterio
from osgeo import gdal
from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsMapLayerProxyModel,
    QgsProject,
    QgsRectangle,
    QgsMapLayerType,
    QgsVectorLayer,
    QgsField,
    QgsGeometry,
    QgsFeature,
    QgsSymbol,
    QgsWkbTypes,
    QgsRuleBasedRenderer,
)
import sip

# QVariant has been deprecated in version 3.38, use QMetaType instead
qgis_version = Qgis.QGIS_VERSION_INT
if qgis_version < 33800:
    from qgis.PyQt.QtCore import QVariant as QMetaType

    QMetaType.QString = QMetaType.String
else:
    from qgis.PyQt.QtCore import QMetaType

from qgis.gui import QgisInterface, QgsFileWidget, QgsMapToolPan
from qgis.PyQt import QtCore
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtGui import QColor, QKeySequence, QPixmap
from qgis.PyQt.QtWidgets import QDockWidget, QFileDialog, QShortcut, QLabel
from rasterio.windows import from_bounds as window_from_bounds

from ..ui import (
    ICON_TYPE,
    DefaultSettings,
    Settings,
    UI_Selector,
    save_user_settings,
)
from .canvasTool import (
    Canvas_Extent,
    Canvas_Points,
    Canvas_Rectangle,
    ClickTool,
    RectangleMapTool,
    SAM_PolygonFeature,
)
from .geoTool import ImageCRSManager
from .messageTool import MessageTool
from .SAM2Tool import SAM2_Model

from ..ui.image import * 

SAM2_Model_Types_Full: List[str] = ["hiera_l", "hiera_t"]
SAM2_Model_Types = [i.split(" ")[0].strip() for i in SAM2_Model_Types_Full]


shp_file_load_filter = (
    "Shapefile (*.shp);;"  # ESRI Shapefile
    "Geopackage (*.gpkg);;"  # GeoPackage
    "SQLite (*.sqlite);;"  # SQLite
    "GeoJSON (*.geojson);;"  # GeoJSON
    "All files (*)"  # All files
)
shp_file_create_filter = (
    "Shapefile (*.shp);;"  # ESRI Shapefile
    "Geopackage (*.gpkg);;"  # GeoPackage
    "SQLite (*.sqlite);;"  # SQLite
    "All files (*)"  # All files
)


class ParseRangeThread(QThread):
    def __init__(
        self,
        retrieve_range: pyqtSignal,
        raster_path: str,
        extent: List[float],
        bands: List[int],
    ):
        super().__init__()
        self.retrieve_range = retrieve_range
        self.raster_path = raster_path
        self.extent = extent
        self.bands = bands

    def run(self):
        with rasterio.open(self.raster_path) as src:
            # if image is too large, downsample it
            width = src.width
            height = src.height

            scale = width * height / 100000000
            if scale >= 2:
                width = int(width / scale)
                height = int(height / scale)
            if self.extent is None:
                window = None
            else:
                window = window_from_bounds(*self.extent, src.transform)

            arr = src.read(
                self.bands, out_shape=(len(self.bands), height, width), window=window
            )
            if src.meta["nodata"] is not None:
                arr = np.ma.masked_equal(arr, src.meta["nodata"])

        self.retrieve_range.emit(f"{np.nanmin(arr)}, {np.nanmax(arr)}")



class ShowPatchExtentThread(QThread):
    """
    A thread for calculating and returning the geographic extents of all crop patches.
    This logic strictly follows the non-overlapping, edge-adaptive GDAL sliding window algorithm.
    """
    def __init__(self, retrieve_patch, raster_path, patch_size):
        super().__init__()
        self.retrieve_patch = retrieve_patch  # Signal for sending results
        self.raster_path = raster_path
        self.patch_size = patch_size

    def run(self):
        extents = []
        
        # Open raster file using GDAL
        poRasterDS = gdal.Open(self.raster_path, gdal.GA_ReadOnly)
        if poRasterDS is None:
            # In practice, it's better to send error information through signals
            print(f"GDAL Error: Unable to open raster file {self.raster_path}")
            self.retrieve_patch.emit("[]")  # Send an empty list to indicate failure
            return

        # Get image dimensions and geotransform parameters
        nCols = poRasterDS.RasterXSize
        nRows = poRasterDS.RasterYSize
        adfGeoTransform = poRasterDS.GetGeoTransform()

        # --- Reproduce the crop patch count calculation logic from sliding_window_predict_with_gdal ---
        quotientCols = nCols // self.patch_size
        residueCols = 1 if nCols % self.patch_size > 0 else 0
        numCols = quotientCols + residueCols

        quotientRows = nRows // self.patch_size
        residueRows = 1 if nRows % self.patch_size > 0 else 0
        numRows = quotientRows + residueRows

        # --- Iterate through all crop patches to calculate their geographic extents ---
        extent_dict = {}
        for j in range(numRows):
            for i in range(numCols):
                # Theoretical top-left pixel coordinates
                current_patch_idx = j * numCols + i
                topLeftX = i * self.patch_size
                topLeftY = j * self.patch_size

                # Check if exceeds right boundary
                if topLeftX + self.patch_size > nCols:
                    adjusted_topLeftX = nCols - self.patch_size
                else:
                    adjusted_topLeftX = topLeftX

                # Check if exceeds bottom boundary
                if topLeftY + self.patch_size > nRows:
                    adjusted_topLeftY = nRows - self.patch_size
                else:
                    adjusted_topLeftY = topLeftY
                
                # Ensure coordinates are not negative
                adjusted_topLeftX = max(0, adjusted_topLeftX)
                adjusted_topLeftY = max(0, adjusted_topLeftY)
                
                # Calculate geographic boundaries of the crop patch
                minx = adfGeoTransform[0] + adjusted_topLeftX * adfGeoTransform[1]
                maxy = adfGeoTransform[3] + adjusted_topLeftY * adfGeoTransform[5]
                
                bottomRightX_pix = adjusted_topLeftX + self.patch_size
                bottomRightY_pix = adjusted_topLeftY + self.patch_size
                
                maxx = adfGeoTransform[0] + bottomRightX_pix * adfGeoTransform[1]
                miny = adfGeoTransform[3] + bottomRightY_pix * adfGeoTransform[5]

                extent_dict[current_patch_idx] = [minx, miny, maxx, maxy]
                extents.append([minx, miny, maxx, maxy])

        poRasterDS = None 

        # Send all calculated extents via signal
        self.retrieve_patch.emit(f"{extents}")


class Selector(QDockWidget):
    execute_SAM = pyqtSignal()
    retrieve_patch = pyqtSignal(str)

    def __init__(self, parent, iface: QgisInterface, cwd: str):
        # super().__init__()
        QDockWidget.__init__(self)
        self.parent = parent
        self.iface = iface
        self.cwd = Path(cwd)
        self.canvas = iface.mapCanvas()
        self.feature_dir = self.cwd
        self.project: QgsProject = QgsProject.instance()
        self.toolPan = QgsMapToolPan(self.canvas)
        self.dockFirstOpen = True
        self.prompt_history: List[str] = []
        self.sam_feature_history: List[List[int]] = []
        self.preview_mode: bool = False
        self.t_area: float = 0.0
        self.max_polygon_mode: bool = False
        self.need_execute_sam_toggle_mode: bool = True
        self.need_execute_sam_filter_area: bool = True
        self.feature_loaded: bool = False
        self.default_name: str = "polygon_sam2_mask"
        self.ds_sampler: None

    def open_widget(self):
        """Create widget selector"""
        # seem not necessary to set True
        # self.parent.toolbar.setVisible(True)
        if self.dockFirstOpen:
            self.crs_project: QgsCoordinateReferenceSystem = self.project.crs()

            if self.receivers(self.execute_SAM) == 0:
                self.execute_SAM.connect(self.execute_segmentation)

            self.wdg_sel = UI_Selector

            # GUI elements
            # pixmap = QPixmap(":/plugins/Geo-SAM2/tools/geo_sam2.png")
            # # pixmap = QPixmap(":/plugins/Geo-SAM2/tools/geo_sam2.png").scaledToWidth(100, Qt.SmoothTransformation)  # Scale to desired width
            
            # self.wdg_sel.label_12.setPixmap(pixmap)
            # self.wdg_sel.label_12.setAlignment(Qt.AlignCenter)
            # self.wdg_sel.label_12.setScaledContents(True)
            # 1. Set original image
            pixmap = QPixmap(":/plugins/Geo-SAM2/ui/image/geo_sam2.png")
            self.wdg_sel.label_12.setPixmap(pixmap)
            self.wdg_sel.label_12.setAlignment(Qt.AlignCenter)
            self.wdg_sel.label_12.setScaledContents(True)

            # Create text label and overlay it on the image label
            text_label = QLabel(self.wdg_sel.label_12)
            text_label.setAlignment(Qt.AlignCenter)  # Ensure text is centered within the label
            text_label.setStyleSheet("""
                QLabel {
                    background: transparent;
                    color: black;
                }
            """)

            # Use HTML format to set multi-line text with different styles
            text_label.setText("""
            <div style="text-align: center;">
                <span style=" font-size:14pt; font-weight:600;">Geo-SAM2</span><br>
                <span style=" font-size:12pt;">Interactive Remote Sensing Segmentation Tool Based on</span><br>
                <span style=" font-size:12pt;">Segment Anything Model 2</span><br>
            </div>
            """)

            # Ensure text label size synchronizes with image label
            def update_text_label_size():
                text_label.setGeometry(0, 0, self.wdg_sel.label_12.width(), self.wdg_sel.label_12.height())

            # Initial size setting
            update_text_label_size()

            # Monitor parent label size changes, dynamically adjust text label size
            self.wdg_sel.label_12.resizeEvent = lambda event: update_text_label_size()

            # Ensure text label is above the image label
            text_label.raise_()
            
            
            ######### Setting default parameters for items #########
            self.wdg_sel.MapLayerComboBox.setFilters(
                QgsMapLayerProxyModel.PolygonLayer | QgsMapLayerProxyModel.VectorLayer
            )
            self.wdg_sel.MapLayerComboBox.setAllowEmptyLayer(True)
            self.wdg_sel.MapLayerComboBox.setAdditionalLayers([None])

            self.wdg_sel.QgsFile_feature.setStorageMode(QgsFileWidget.GetDirectory)

            # set button checkable
            self.wdg_sel.pushButton_fg.setCheckable(True)
            self.wdg_sel.pushButton_bg.setCheckable(True)
            self.wdg_sel.pushButton_rect.setCheckable(True)

            self.wdg_sel.pushButton_reset_settings.clicked.connect(
                self.reset_default_settings
            )
                  
            self.wdg_sel.checkBox_show_boundary.toggled.connect(
                self.toggle_encoding_extent
            )

            ########## connect function to widget items ##########
            self.wdg_sel.pushButton_fg.clicked.connect(self.draw_foreground_point)
            self.wdg_sel.pushButton_bg.clicked.connect(self.draw_background_point)
            self.wdg_sel.pushButton_rect.clicked.connect(self.draw_rect)

            # tools
            self.wdg_sel.pushButton_clear.clicked.connect(self.clear_layers)
            self.wdg_sel.pushButton_undo.clicked.connect(self.undo_last_prompt)
            self.wdg_sel.pushButton_save.clicked.connect(self.save_shp_file)

            self.wdg_sel.MapLayerComboBox.layerChanged.connect(self.set_vector_layer)
            self.wdg_sel.pushButton_load_file.clicked.connect(self.load_vector_file)
            self.wdg_sel.pushButton_create_file.clicked.connect(self.create_vector_file)

            self.wdg_sel.pushButton_load_feature.clicked.connect(self.load_feature)
            
            self.wdg_sel.checkBox_show_crop_lines.setChecked(False)
            self.wdg_sel.checkBox_show_crop_lines.toggled.connect(self.toggle_show_crop_lines)
            self.retrieve_patch.connect(self.show_patch_extent_in_canvas)
            
            self.wdg_sel.pushButton_zoom_extent.clicked.connect(self.zoom_to_extent)
            self.wdg_sel.radioButton_enable.setChecked(False)
            self.wdg_sel.radioButton_enable.toggled.connect(self.toggle_edit_mode)

            self.wdg_sel.radioButton_exe_hover.setChecked(False)
            self.wdg_sel.radioButton_exe_hover.toggled.connect(
                self.toggle_sam_hover_mode
            )

            # threshold of area
            self.wdg_sel.Box_min_pixel.valueChanged.connect(self.filter_feature_by_area)

            # only keep max object mode
            self.wdg_sel.radioButton_max_polygon_mode.toggled.connect(
                self.toggle_max_polygon_mode
            )

            self.wdg_sel.ColorButton_bgpt.colorChanged.connect(self.reset_points_bg)
            self.wdg_sel.ColorButton_fgpt.colorChanged.connect(self.reset_points_fg)
            self.wdg_sel.ColorButton_bbox.colorChanged.connect(
                self.reset_rectangular_color
            )
            self.wdg_sel.ColorButton_extent.colorChanged.connect(
                self.reset_extent_color
            )
            self.wdg_sel.ColorButton_prompt.colorChanged.connect(
                self.reset_prompt_polygon_color
            )
            self.wdg_sel.ColorButton_preview.colorChanged.connect(
                self.reset_preview_polygon_color
            )

            self.wdg_sel.SpinBoxPtSize.valueChanged.connect(self.reset_points_size)
            # self.wdg_sel.comboBoxIconType.clear()
            self.wdg_sel.comboBoxIconType.addItems(list(ICON_TYPE.keys()))
            # self.wdg_sel.comboBoxIconType.setCurrentText("Circle")
            self.wdg_sel.comboBoxIconType.currentTextChanged.connect(
                self.reset_points_icon
            )

            # If a signal is connected to several slots,
            # the slots are activated in the same order in which the connections were made, when the signal is emitted.
            self.wdg_sel.closed.connect(self.destruct)
            self.wdg_sel.closed.connect(self.iface.actionPan().trigger)
            self.wdg_sel.closed.connect(self.reset_to_project_crs)

            ########### shortcuts ############
            # create shortcuts
            self.shortcut_clear = QShortcut(QKeySequence(Qt.Key_C), self.wdg_sel)
            self.shortcut_undo = QShortcut(QKeySequence(Qt.Key_Z), self.wdg_sel)
            self.shortcut_save = QShortcut(QKeySequence(Qt.Key_S), self.wdg_sel)
            self.shortcut_hover_mode = QShortcut(QKeySequence(Qt.Key_P), self.wdg_sel)
            self.shortcut_tab = QShortcut(QKeySequence(Qt.Key_Tab), self.wdg_sel)
            self.shortcut_undo_sam_pg = QShortcut(
                QKeySequence(QKeySequence.Undo), self.wdg_sel
            )

            # connect shortcuts
            self.shortcut_clear.activated.connect(self.clear_layers)
            self.shortcut_undo.activated.connect(self.undo_last_prompt)
            self.shortcut_save.activated.connect(self.save_shp_file)
            self.shortcut_hover_mode.activated.connect(self.toggle_hover_mode)
            self.shortcut_tab.activated.connect(self.loop_prompt_type)
            self.shortcut_undo_sam_pg.activated.connect(self.undo_sam_polygon)

            # set context for shortcuts to application
            # this will make shortcuts work even if the widget is not focused
            self.shortcut_clear.setContext(Qt.ApplicationShortcut)
            self.shortcut_undo.setContext(Qt.ApplicationShortcut)
            self.shortcut_save.setContext(Qt.ApplicationShortcut)
            self.shortcut_hover_mode.setContext(Qt.ApplicationShortcut)
            self.shortcut_tab.setContext(Qt.ApplicationShortcut)
            self.shortcut_undo_sam_pg.setContext(Qt.ApplicationShortcut)

            # disable tool buttons when no feature loaded
            self.wdg_sel.radioButton_enable.setChecked(False)
            ########## set default Settings ##########
            self.set_user_settings()

            ########## set dock ##########
            self.wdg_sel.setFloating(True)
            self.wdg_sel.setFocusPolicy(Qt.StrongFocus)

            # default is fgpt, but do not change when reloading feature folder
            # self.reset_prompt_type()
            self.dockFirstOpen = False
        else:
            self.clear_layers(clear_extent=True)

        # add widget to QGIS
        # self.iface.addDockWidget(Qt.TopDockWidgetArea, self.wdg_sel)
        self.iface.addDockWidget(Qt.RightDockWidgetArea, self.wdg_sel)
        

        self.toggle_edit_mode()
        # self.toggle_encoding_extent()

        # if not self.wdg_sel.isUserVisible():
        #     self.wdg_sel.setUserVisible(True)

    def alpha_color(self, color: QColor, alpha: float) -> QColor:
        return QColor(color.red(), color.green(), color.blue(), alpha)

    def on_threshold_changed(self):
        """Called when NMS threshold is changed"""
        if not hasattr(self, 'sam2_model') or not self.sam2_model.raw_box_results:
            return
        
        nms_value = self.wdg_sel.spinBox_nmsThreshold.value()
        score_value = self.wdg_sel.spinBox_classThreshold.value()

        # Update display with new threshold
        self.sam2_model.filter_and_display_results(self, nms_value, score_value)

    def set_styles_settings(self, settings):
        self.wdg_sel.ColorButton_bgpt.setColor(QColor(settings["bg_color"]))
        self.wdg_sel.ColorButton_fgpt.setColor(QColor(settings["fg_color"]))
        self.wdg_sel.ColorButton_bbox.setColor(QColor(settings["bbox_color"]))
        self.wdg_sel.ColorButton_extent.setColor(QColor(settings["extent_color"]))
        self.wdg_sel.ColorButton_prompt.setColor(QColor(settings["prompt_color"]))
        self.wdg_sel.ColorButton_preview.setColor(QColor(settings["preview_color"]))

        self.wdg_sel.SpinBoxPtSize.setValue(settings["pt_size"])

        self.wdg_sel.comboBoxIconType.setCurrentText(settings["icon_type"])
        # colors
        self.style_preview_polygon: Dict[str, Any] = {
            "line_color": QColor(Settings["preview_color"]),
            "fill_color": self.alpha_color(QColor(Settings["preview_color"]), 10),
            "line_width": 2,
        }
        self.style_prompt_polygon: Dict[str, Any] = {
            "line_color": QColor(Settings["prompt_color"]),
            "fill_color": self.alpha_color(QColor(Settings["prompt_color"]), 10),
            "line_width": 3,
        }

    def set_user_settings(self):
        MessageTool.MessageLog(f"user setting: {Settings}")
        self.set_styles_settings(Settings)

        if Settings["max_polygon_only"]:
            self.wdg_sel.radioButton_max_polygon_mode.setChecked(True)
            MessageTool.MessageLog("Max object mode on")
        else:
            self.wdg_sel.radioButton_max_polygon_mode.setChecked(False)
            MessageTool.MessageLog("Max object mode off")

    def reset_default_settings(self):
        save_user_settings({}, mode="overwrite")
        if not DefaultSettings["max_polygon_only"]:
            self.wdg_sel.radioButton_max_polygon_mode.setChecked(False)

        self.set_styles_settings(DefaultSettings)
    
    def show_patch_extent_in_canvas(self, extents: str):
        extents = eval(extents)
        if not extents:
            return

        # Check if layer named "Patch_Extents" already exists
        existing_layers = QgsProject.instance().mapLayersByName("Patch_Extents")
        
        if existing_layers:
            # If exists, use the first found layer
            vl = existing_layers[0]
            # Delete all existing features
            vl.startEditing()
            all_feature_ids = [f.id() for f in vl.getFeatures()]
            if all_feature_ids:
                vl.deleteFeatures(all_feature_ids)
            vl.commitChanges()
        else:
            # If not exists, create new layer
            vl = QgsVectorLayer("Polygon?crs=" + self.iface.mapCanvas().mapSettings().destinationCrs().authid(), 
                                "Patch_Extents", "memory")
            pr = vl.dataProvider()
            pr.addAttributes([QgsField("patch_id", QMetaType.Int), QgsField("highlight", QMetaType.Int)])
            vl.updateFields()
            QgsProject.instance().addMapLayer(vl)

        # Get data provider
        pr = vl.dataProvider()
        
        # Prepare features
        features = []
        num_patch = len(extents)
        idx = np.random.randint(0, num_patch, size=(int(num_patch / 10)))
        highlight_set = set(idx)
        highlight_set.add(num_patch - 1)  # always highlight last

        for i, extent in enumerate(extents):
            minx, miny, maxx, maxy = extent
            rect = QgsRectangle(minx, miny, maxx, maxy)
            geom = QgsGeometry.fromRect(rect)
            feat = QgsFeature()
            feat.setGeometry(geom)
            feat.setAttributes([i, 1 if i in highlight_set else 0])
            features.append(feat)

        # Add features
        vl.startEditing()
        pr.addFeatures(features)
        vl.commitChanges()
        vl.updateExtents()

        # Set style (only needed when creating new layer)
        if not existing_layers:
            symbol = QgsSymbol.defaultSymbol(QgsWkbTypes.PolygonGeometry)
            symbol.setOpacity(0.6)  # Overall transparency
            symbol.symbolLayer(0).setStrokeColor(QColor(255, 0, 0))
            symbol.symbolLayer(0).setBrushStyle(Qt.NoBrush)  # No fill

            # Use Rule-based renderer to distinguish normal and highlighted
            root_rule = QgsRuleBasedRenderer.Rule(None)
            renderer = QgsRuleBasedRenderer(root_rule)

            # Highlight rule
            rule_highlight = QgsRuleBasedRenderer.Rule(
                symbol.clone(),
                label='Random Highlighted Patches'
            )
            rule_highlight.setFilterExpression('"highlight" = 1')  # ✅ Correct setting method
            rule_highlight.symbol().symbolLayer(0).setStrokeWidth(1)
            root_rule.appendChild(rule_highlight)

            # Normal rule
            rule_normal = QgsRuleBasedRenderer.Rule(
                symbol.clone(),
                label='Normal Patches'
            )
            rule_normal.setFilterExpression('"highlight" = 0')  # ✅ Correct setting method
            rule_normal.symbol().symbolLayer(0).setStrokeWidth(0.2)
            root_rule.appendChild(rule_normal)

            vl.setRenderer(renderer)

        self.patch_extent_layer = vl  # Save reference for easy deletion later

    def toggle_show_crop_lines(self):
        """Display all crop patch extents in canvas according to GDAL sliding window logic"""
        if self.wdg_sel.checkBox_show_crop_lines.isChecked():
            self.wdg_sel.checkBox_show_boundary.setChecked(False)
            if self.feature_loaded:

                self.feature_dir = self.wdg_sel.QgsFile_feature.filePath()
                if self.feature_dir == "":
                    MessageTool.MessageLog(f"Input Feature Folder is None.")
                    return

                param_file = os.path.join(self.feature_dir, "sam2_encoder_parameters.json")
                if not os.path.exists(param_file):
                    MessageTool.MessageLog(f"{param_file} is not exist.")
                    return

                # Read JSON file to get input raster path
                with open(param_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                input_path = data["inputs"]["INPUT"]
                if not os.path.exists(input_path):
                    MessageTool.MessageBar(
                        "Error",
                        f"Raster file does not exist: {input_path}",
                        duration=30,
                    )
                    return

                # Define crop patch size, consistent with GDAL processing
                patch_size = 1024  # SAM2 model default size

                # Start a new thread to calculate all crop extents, avoid UI freezing
                self.show_patch_extent_thread = ShowPatchExtentThread(
                    self.retrieve_patch, input_path, patch_size
                )
                self.show_patch_extent_thread.start()
        else:
            # Clear displayed crop lines, this part logic remains unchanged
            if hasattr(self, 'patch_extent_layer'):
                QgsProject.instance().removeMapLayer(self.patch_extent_layer.id())
                del self.patch_extent_layer
                
            self.canvas_extent.clear()
            self.clear_layers()
            
    def disconnect_safely(self, item):
        try:
            item.disconnect()
        except:
            pass

    def reset_to_project_crs(self):
        if self.crs_project != self.project.crs():
            MessageTool.MessageBar(
                "Note:", "Project CRS has been reset to original CRS."
            )
            self.project.setCrs(self.crs_project)

    def destruct(self):
        """Destruct actions when closed widget"""
        self.clear_layers(clear_extent=True)
        self.reset_to_project_crs()
        self.iface.actionPan().trigger()

    def unload(self):
        """Unload actions when plugin is closed"""
        self.clear_layers(clear_extent=True)
        if hasattr(self, "shortcut_tab"):
            self.disconnect_safely(self.shortcut_tab)
        if hasattr(self, "shortcut_undo_sam_pg"):
            self.disconnect_safely(self.shortcut_undo_sam_pg)
        if hasattr(self, "shortcut_clear"):
            self.disconnect_safely(self.shortcut_clear)
        if hasattr(self, "shortcut_undo"):
            self.disconnect_safely(self.shortcut_undo)
        if hasattr(self, "shortcut_save"):
            self.disconnect_safely(self.shortcut_save)
        if hasattr(self, "shortcut_hover_mode"):
            self.disconnect_safely(self.shortcut_hover_mode)
        if hasattr(self, "wdg_sel"):
            self.disconnect_safely(self.wdg_sel.MapLayerComboBox.layerChanged)
            self.iface.removeDockWidget(self.wdg_sel)
        self.destruct()

    def topping_polygon_sam_layer(self):
        """Move polygon layer of SAM result to top of TOC"""
        root = QgsProject.instance().layerTreeRoot()
        tree_layer = root.findLayer(self.polygon.layer.id())

        if tree_layer is None:
            return None
        if not tree_layer.isVisible():
            tree_layer.setItemVisibilityChecked(True)
        if root.children()[0] == tree_layer:
            return None

        # move to top
        tl_clone = tree_layer.clone()
        root.insertChildNode(0, tl_clone)
        parent_tree_layer = tree_layer.parent()
        parent_tree_layer.removeChildNode(tree_layer)

    def clear_canvas_layers_safely(self, clear_extent: bool = False):
        """Clear canvas layers safely"""
        self.canvas.refresh()
        if hasattr(self, "canvas_points"):
            self.canvas_points.clear()
        if hasattr(self, "canvas_rect"):
            self.canvas_rect.clear()
        if hasattr(self, "canvas_extent") and clear_extent:
            self.canvas_extent.clear()
        if hasattr(self, "polygon"):
            self.polygon.clear_canvas_polygons()
        self.canvas.refresh()

    def clear_preview_prompt_polygon(self):
        self.tool_click_fg.clear_hover_prompt()
        self.tool_click_bg.clear_hover_prompt()
        self.tool_click_rect.clear_hover_prompt()
        self.polygon.canvas_preview_polygon.clear()

    def _set_feature_related(self):
        """Init or reload feature related objects"""
        # init feature related objects
        self.sam2_model = SAM2_Model(self.feature_dir, str(self.cwd))
        MessageTool.MessageLog(
            f"SAM2 Image Encoder Features with {self.sam2_model.feature_size} patches in '{Path(self.feature_dir).name}' have been loaded, you can start labeling now"
        )

        self.res = self.sam2_model.res

        self.img_crs_manager = ImageCRSManager(self.sam2_model.img_crs)
        self.canvas_points = Canvas_Points(self.canvas, self.img_crs_manager)
        self.canvas_rect = Canvas_Rectangle(self.canvas, self.img_crs_manager)
        self.canvas_extent = Canvas_Extent(self.canvas, self.img_crs_manager)

        # reset canvas extent
        self.sam_extent_canvas_crs = self.img_crs_manager.img_extent_to_crs(
            self.sam2_model.extent, QgsProject.instance().crs()
        )
        
        # self.canvas.setExtent(self.sam_extent_canvas_crs)
        # self.canvas.refresh()

        # init tools
        self.tool_click_fg = ClickTool(
            self.canvas,
            self.canvas_points,
            "fgpt",
            self.prompt_history,
            self.execute_SAM,
        )
        self.tool_click_bg = ClickTool(
            self.canvas,
            self.canvas_points,
            "bgpt",
            self.prompt_history,
            self.execute_SAM,
        )
        self.tool_click_rect = RectangleMapTool(
            self.canvas_rect,
            self.prompt_history,
            self.execute_SAM,
            self.img_crs_manager,
        )

        # New: Set right-click save functionality for tool classes
        self.tool_click_fg.right_click_save = self.handle_right_click_save
        self.tool_click_bg.right_click_save = self.handle_right_click_save
        self.tool_click_rect.right_click_save = self.handle_right_click_save

        self.reset_all_styles()

    # def handle_right_click_save(self):
    #     """Handle right-click save functionality"""
    #     if self.preview_mode:
    #         MessageTool.MessageLog("Right-click: Execute save operation")
    #         self.save_shp_file()
    #         return True
    #     return False
    def handle_right_click_save(self):
        """Handle right-click save functionality"""
        # if self.preview_mode:
        MessageTool.MessageLog("Right-click: Execute save operation")
        self.save_shp_file()
        return True
        # return False

    def zoom_to_extent(self):
        """Change Canvas extent to feature extent"""
        if hasattr(self, "sam_extent_canvas_crs"):
            self.canvas.setExtent(self.sam_extent_canvas_crs)
            self.canvas.refresh()

    def loop_prompt_type(self):
        """Loop prompt type"""
        # reset pressed to False before loop
        self.tool_click_fg.pressed = False
        self.tool_click_bg.pressed = False
        self.tool_click_rect.pressed = False

        self.clear_preview_prompt_polygon()

        if self.wdg_sel.pushButton_fg.isChecked():
            self.draw_background_point()
        elif self.wdg_sel.pushButton_bg.isChecked():
            self.draw_rect()
        elif self.wdg_sel.pushButton_rect.isChecked():
            self.draw_foreground_point()

    def undo_last_prompt(self):
        if len(self.prompt_history) > 0:
            prompt_last = self.prompt_history.pop()
            if prompt_last == "bbox":
                # self.canvas_rect.clear()
                self.canvas_rect.popRect()
            else:
                self.canvas_points.popPoint()
            self.execute_SAM.emit()

    def toggle_edit_mode(self):
        """Enable or disable the widget selector"""
        # radioButton = self.sender()
        radioButton = self.wdg_sel.radioButton_enable
        if not radioButton.isChecked():
            self.canvas.setMapTool(self.toolPan)
            
            if self.wdg_sel.pushButton_bg.isChecked():
                self.wdg_sel.pushButton_bg.toggle()
            if self.wdg_sel.pushButton_rect.isChecked():
                self.wdg_sel.pushButton_rect.toggle()
            if self.wdg_sel.pushButton_fg.isChecked():
                self.wdg_sel.pushButton_fg.toggle()
            
            self.wdg_sel.pushButton_fg.setEnabled(False)
            self.wdg_sel.pushButton_bg.setEnabled(False)
            self.wdg_sel.pushButton_rect.setEnabled(False)
            self.wdg_sel.pushButton_clear.setEnabled(False)
            self.wdg_sel.pushButton_undo.setEnabled(False)
            self.wdg_sel.pushButton_save.setEnabled(False)
            self.wdg_sel.Box_min_pixel.setEnabled(False)
            self.wdg_sel.radioButton_exe_hover.setEnabled(False)
            self.wdg_sel.radioButton_exe_hover.setChecked(False)
            self.wdg_sel.radioButton_max_polygon_mode.setEnabled(False)
            
        else:
            self.wdg_sel.pushButton_fg.setEnabled(True)
            self.wdg_sel.pushButton_bg.setEnabled(True)
            self.wdg_sel.pushButton_rect.setEnabled(True)
            self.wdg_sel.pushButton_clear.setEnabled(True)
            self.wdg_sel.pushButton_undo.setEnabled(True)
            self.wdg_sel.pushButton_save.setEnabled(True)
            self.wdg_sel.Box_min_pixel.setEnabled(True)
            self.wdg_sel.radioButton_exe_hover.setEnabled(True)
            self.wdg_sel.radioButton_max_polygon_mode.setEnabled(True)

    def toggle_encoding_extent(self):
        """Show or hide extent of SAM2 encoded feature"""
        if self.wdg_sel.checkBox_show_boundary.isChecked():
            self.wdg_sel.checkBox_show_crop_lines.setChecked(False)
            if self.feature_loaded:
                self.canvas_extent.add_extent(self.sam_extent_canvas_crs)
            else:
                return None
            show_extent = True
        else:
            if not hasattr(self, "canvas_extent"):
                return None
            self.canvas_extent.clear()
            show_extent = False
        save_user_settings({"show_boundary": show_extent}, mode="update")

    def toggle_hover_mode(self):
        """Toggle move mode in widget selector."""
        if self.wdg_sel.radioButton_exe_hover.isChecked():
            self.wdg_sel.radioButton_exe_hover.setChecked(False)
            self.need_execute_sam_toggle_mode = True
        else:
            self.wdg_sel.radioButton_exe_hover.setChecked(True)
            self.need_execute_sam_toggle_mode = False
        # toggle move mode in sam model
        self.toggle_sam_hover_mode()

    def toggle_sam_hover_mode(self):
        """Toggle move mode in sam model"""
        if self.wdg_sel.radioButton_exe_hover.isChecked():
            self.preview_mode = True
            self.tool_click_fg.preview_mode = True
            self.tool_click_bg.preview_mode = True
            self.tool_click_rect.preview_mode = True
        else:
            self.preview_mode = False
            self.tool_click_fg.preview_mode = False
            self.tool_click_bg.preview_mode = False
            self.tool_click_rect.preview_mode = False
            # clear hover prompts
            self.clear_preview_prompt_polygon()

        if self.need_execute_sam_toggle_mode:
            self.execute_SAM.emit()

    def toggle_max_polygon_mode(self):
        if self.wdg_sel.radioButton_max_polygon_mode.isChecked():
            self.max_polygon_mode = True
            self.wdg_sel.Box_min_pixel.setEnabled(False)
        else:
            self.max_polygon_mode = False
            self.wdg_sel.Box_min_pixel.setEnabled(True)
        save_user_settings({"max_polygon_only": self.max_polygon_mode}, mode="update")

        if self.feature_loaded:
            self.execute_SAM.emit()

    def is_pressed_prompt(self):
        """Check if the prompt is clicked or hovered"""
        if (
            self.tool_click_fg.pressed
            or self.tool_click_bg.pressed
            or self.tool_click_rect.pressed
        ):
            return True
        return False

    def filter_feature_by_area(self):
        """Filter feature by area"""
        if not self.need_execute_sam_filter_area or not hasattr(self, "res"):
            return None

        t_area = self.wdg_sel.Box_min_pixel.value() * self.res**2
        if not hasattr(self, "polygon"):
            return None

        # clear SAM canvas result
        self.polygon.canvas_prompt_polygon.clear()
        if self.preview_mode:
            self.polygon.canvas_preview_polygon.clear()

        # filter feature by new area, only show in prompt canvas
        self.t_area = t_area
        self.polygon.add_geojson_feature_to_canvas(
            self.polygon.geojson_canvas_prompt,
            self,
            target="prompt",
        )

    def ensure_polygon_sam_exist(self, skip_init: bool = False):
        has_vector_layer = False
        for layer in QgsProject.instance().mapLayers().values():
            if layer.type() == QgsMapLayerType.VectorLayer:
                has_vector_layer = True
                break

        if has_vector_layer and hasattr(self, "polygon") and self.polygon is not None:
            # Check if self.polygon.layer exists and hasn't been deleted by C++
            layer = getattr(self.polygon, 'layer', None)
            if layer is not None:
                # ✅ Key: First check if it's deleted using sip.isdeleted()
                if not sip.isdeleted(layer):
                    try:
                        # Then try to safely access attributes
                        if layer.name() == self.default_name:
                            polygon_layer = QgsProject.instance().mapLayer(self.polygon.layer_id)
                            if polygon_layer is not None and not sip.isdeleted(polygon_layer):
                                self.wdg_sel.MapLayerComboBox.setLayer(layer)
                                return None
                    except RuntimeError:
                        # Defensive programming: if still errors, treat as invalid
                        pass

        self.set_vector_layer(reset=True, skip_init=skip_init)

    def execute_segmentation(self) -> bool:
        # check prompt inside feature extent and add last id to history for new prompt
        MessageTool.MessageBar("Info: ", "execute_segmentation")
        self.default_name = "polygon_sam2_mask"
        if len(self.prompt_history) > 0 and self.is_pressed_prompt():
            prompt_last = self.prompt_history[-1]
            if prompt_last == "bbox":
                last_rect = self.canvas_rect.extent
                last_prompt = QgsRectangle(
                    last_rect[0], last_rect[2], last_rect[1], last_rect[3]
                )
            else:
                last_point = self.canvas_points.img_crs_points[-1]
                last_prompt = QgsRectangle(last_point, last_point)
            if not last_prompt.intersects(self.sam2_model.extent):
                self.check_message_box_outside()
                self.undo_last_prompt()
                return False

            self.ensure_polygon_sam_exist()

            # add last id to history
            features = list(self.polygon.layer.getFeatures())
            if len(list(features)) == 0:
                last_id = 1
            else:
                last_id = features[-1].id() + 1

            if (
                len(self.sam_feature_history) >= 1
                and len(self.sam_feature_history[-1]) == 1
            ):
                self.sam_feature_history[-1][0] = last_id
            else:
                self.sam_feature_history.append([last_id])

        self.ensure_polygon_sam_exist()

        # clear canvas prompt polygon for new prompt
        if self.is_pressed_prompt():
            self.polygon.canvas_prompt_polygon.clear()

        # execute segmentation
        if not self.sam2_model.sam2_predict(self):
            # out of extent and not in preview mode
            self.undo_last_prompt()

        # show pressed prompt result in preview mode
        if self.preview_mode and self.is_pressed_prompt():
            self.polygon.add_geojson_feature_to_canvas(
                self.polygon.geojson_canvas_preview,  # update with canvas polygon
                self,
                target="prompt",
                overwrite_geojson=True,
            )
        self.topping_polygon_sam_layer()

        return True
               
    def draw_foreground_point(self):
        """draw foreground point in canvas"""
        if not hasattr(self, "tool_click_fg"):
            MessageTool.MessageBar("Oops: ", "Please load feature folder first")
            return None
        self.canvas.setMapTool(self.tool_click_fg)
        button = self.wdg_sel.pushButton_fg
        if not button.isChecked():
            button.toggle()

        if self.wdg_sel.pushButton_bg.isChecked():
            self.wdg_sel.pushButton_bg.toggle()
        if self.wdg_sel.pushButton_rect.isChecked():
            self.wdg_sel.pushButton_rect.toggle()
        self.prompt_type = "fgpt"

    def draw_background_point(self):
        """draw background point in canvas"""
        if not hasattr(self, "tool_click_bg"):
            MessageTool.MessageBar("Oops: ", "Please load feature folder first")
            return None
        self.canvas.setMapTool(self.tool_click_bg)
        button = self.wdg_sel.pushButton_bg
        if not button.isChecked():
            button.toggle()

        if self.wdg_sel.pushButton_fg.isChecked():
            self.wdg_sel.pushButton_fg.toggle()
        if self.wdg_sel.pushButton_rect.isChecked():
            self.wdg_sel.pushButton_rect.toggle()
        self.prompt_type = "bgpt"

    def draw_rect(self):
        """draw rectangle in canvas"""
        if not hasattr(self, "tool_click_rect"):
            MessageTool.MessageBar("Oops: ", "Please load feature folder first")
            return None
        self.canvas.setMapTool(self.tool_click_rect)
        button = self.wdg_sel.pushButton_rect  # self.sender()
        if not button.isChecked():
            button.toggle()

        if self.wdg_sel.pushButton_fg.isChecked():
            self.wdg_sel.pushButton_fg.toggle()
        if self.wdg_sel.pushButton_bg.isChecked():
            self.wdg_sel.pushButton_bg.toggle()
        self.prompt_type = "bbox"

    def only_draw_rect(self):
        """draw rectangle in canvas"""
        if not hasattr(self, "tool_click_rect"):
            MessageTool.MessageBar("Oops: ", "Please load feature folder first")
            return None

        self.tool_click_rect.should_emit_signal = False
        self.canvas.setMapTool(self.tool_click_rect)

        button = self.wdg_sel.pushButton_add_rect  # self.sender()
        if not button.isChecked():
            button.toggle()

    @QtCore.pyqtSlot()  # add descriptor to ignore the input parameter from trigger
    def set_vector_layer(self, reset: bool = False, skip_init: bool=False) -> None:
        """set sam2-oil output vector layer"""
        new_layer = None# noqa: F821
        
        kwargs_preview_polygon=self.style_preview_polygon
        kwargs_prompt_polygon=self.style_prompt_polygon

        if reset:
            MessageTool.MessageLog(f"reset: {reset}")
            self.polygon = SAM_PolygonFeature(
                self.img_crs_manager,
                layer=new_layer,
                kwargs_preview_polygon=kwargs_preview_polygon,
                kwargs_prompt_polygon=kwargs_prompt_polygon,
                default_name=self.default_name,
                skip_init = skip_init,
            )
        else:
            # MessageTool.MessageLog(f"ensure_polygon_sam_exist is None")
            return None
        try:
            if QgsProject.instance().mapLayer(self.polygon.layer_id) is not None:
                self.wdg_sel.MapLayerComboBox.setLayer(self.polygon.layer)
        except Exception as e:
            MessageTool.MessageLog(f"Error: {e}")

    def _process_vector_file(self, file_path: Path, overwrite: bool = False) -> None:
        """Process (load/create) vector file for SAM output"""
        # Check if layer already exists in project
        layer_list = QgsProject.instance().mapLayersByName(file_path.stem)
        if len(layer_list) > 0 and not overwrite:
            self.polygon = SAM_PolygonFeature(
                self.img_crs_manager,
                layer=layer_list[0],
                kwargs_preview_polygon=self.style_preview_polygon,
                kwargs_prompt_polygon=self.style_prompt_polygon,
            )
            if not hasattr(self.polygon, "layer"):
                return None
            MessageTool.MessageBar(
                "Attention",
                f"Layer '{file_path.name}' has already been in the project, "
                "you can start labeling now",
            )
            self.wdg_sel.MapLayerComboBox.setLayer(self.polygon.layer)
        else:
            # Create new layer or load existing file
            self.polygon = SAM_PolygonFeature(
                self.img_crs_manager,
                shapefile=file_path,
                kwargs_preview_polygon=self.style_preview_polygon,
                kwargs_prompt_polygon=self.style_prompt_polygon,
                overwrite=overwrite,
            )
            if not hasattr(self.polygon, "layer"):
                return None
        # clear layer history
        self.sam_feature_history = []
        self.wdg_sel.MapLayerComboBox.setLayer(self.polygon.layer)
        # self.set_user_settings_color()

    def create_vector_file(self) -> None:
        """Create a new vector file for SAM output"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setOption(QFileDialog.DontConfirmOverwrite, True)

        file_path, _ = file_dialog.getSaveFileName(
            None, "Create Vector File", "", shp_file_create_filter
        )
        # If user canceled selection, return
        if file_path is None or file_path == "":
            return None

        file_path = Path(file_path)
        if not file_path.parent.is_dir():
            MessageTool.MessageBoxOK(
                "Oops: Failed to open file, please choose an existing folder"
            )
            return None
        else:
            self._process_vector_file(file_path, overwrite=True)

    def load_vector_file(self) -> None:
        """Load a existed vector file for SAM output"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        file_path, _ = file_dialog.getOpenFileName(
            None, "Load Vector File", "", shp_file_load_filter
        )
        # If user canceled selection, return
        if file_path is None or file_path == "":
            return None

        file_path = Path(file_path)
        if not file_path.parent.is_dir():
            MessageTool.MessageBoxOK(
                "Oops: Failed to open file, please choose an existing folder"
            )
            return None
        else:
            self._process_vector_file(file_path)

    def load_feature(self):
        """load encoded image feature"""
        self.feature_dir = self.wdg_sel.QgsFile_feature.filePath()
        if self.feature_dir is not None and os.path.exists(self.feature_dir):
            self.clear_layers(clear_extent=True)
            self._set_feature_related()
            # self.wdg_sel.radioButton_enable.setChecked(True)
            self.feature_loaded = True
            self.wdg_sel.checkBox_show_crop_lines.setChecked(True) # This line must be after self.feature_loaded = True
        else:
            MessageTool.MessageBar(
                "Oops", "Feature folder not exist, please choose a another folder"
            )

    def clear_layers(self, clear_extent: bool = False):
        """Clear all temporary layers (canvas and new sam result) and reset prompt"""
        self.clear_canvas_layers_safely(clear_extent=clear_extent)
        if hasattr(self, "polygon"):
            self.polygon.clear_canvas_polygons()
        self.prompt_history.clear()

    def save_shp_file(self):
        """save sam result into shapefile layer"""
        need_toggle = False
        if self.preview_mode:
            need_toggle = True
            self.toggle_hover_mode()
            if len(self.prompt_history) == 0:
                MessageTool.MessageBoxOK(
                    "Preview mode only shows the preview of prompts. Click first to apply the prompt."
                )
                self.toggle_hover_mode()
                return False

        if hasattr(self, "polygon"):
            self.polygon.add_geojson_feature_to_layer(
                self.polygon.geojson_canvas_prompt,
                self.t_area,
                self.prompt_history,
                self.max_polygon_mode,
            )

            self.polygon.commit_changes()
            self.polygon.canvas_preview_polygon.clear()
            self.polygon.canvas_prompt_polygon.clear()

            # add last id of new features to history
            features = list(self.polygon.layer.getFeatures())
            if len(list(features)) == 0:
                return None
            last_id = features[-1].id()

            if len(self.sam_feature_history) > 0:
                if self.sam_feature_history[-1][0] > last_id:
                    MessageTool.MessageLog(
                        "New features id is smaller than last id in history",
                        level="warning",
                    )
                self.sam_feature_history[-1].append(last_id)

        # reenable preview mode
        if need_toggle:
            self.toggle_hover_mode()
        self.clear_canvas_layers_safely()
        self.prompt_history.clear()
        self.polygon.reset_geojson()

        # avoid execute sam when reset min pixel to default value
        self.need_execute_sam_filter_area = True

    def reset_prompt_type(self):
        """reset prompt type"""
        if hasattr(self, "prompt_type"):
            if self.prompt_type == "bbox":
                self.draw_rect()
            else:
                self.draw_foreground_point()
        else:
            self.draw_foreground_point()

    def undo_sam_polygon(self):
        """undo last sam polygon"""
        if len(self.sam_feature_history) == 0:
            return None
        last_ids = self.sam_feature_history.pop(-1)
        if len(last_ids) == 1:
            self.clear_layers(clear_extent=False)
            return None
        rm_ids = list(range(last_ids[0], last_ids[1] + 1))
        self.polygon.layer.dataProvider().deleteFeatures(rm_ids)

        # If caching is enabled, a simple canvas refresh might not be sufficient
        # to trigger a redraw and must clear the cached image for the layer
        if self.canvas.isCachingEnabled():
            self.polygon.layer.triggerRepaint()
        else:
            self.canvas.refresh()

    def check_message_box_outside(self):
        if self.preview_mode:
            return True
        else:
            return MessageTool.MessageBoxOK(
                "Point/rectangle is located outside of the feature boundary, click OK to undo last prompt."
            )

    def reset_points_fg(self):
        """Reset point prompt style"""
        fg_color = self.wdg_sel.ColorButton_fgpt.color()
        save_user_settings(
            {
                "fg_color": fg_color.name(),
            },
            mode="update",
        )
        if hasattr(self, "canvas_points"):
            self.canvas_points.foreground_color = fg_color
            self.canvas_points.flush_points_style()
        if hasattr(self, "tool_click_fg"):
            self.tool_click_fg.reset_cursor_color(fg_color.name())

    def reset_points_bg(self):
        """Reset point prompt style"""
        bg_color = self.wdg_sel.ColorButton_bgpt.color()
        save_user_settings(
            {
                "bg_color": bg_color.name(),
            },
            mode="update",
        )
        if hasattr(self, "canvas_points"):
            self.canvas_points.background_color = bg_color
            self.canvas_points.flush_points_style()
        if hasattr(self, "tool_click_bg"):
            self.tool_click_bg.reset_cursor_color(bg_color.name())

    def reset_points_size(self):
        """Reset point prompt style"""
        pt_size = self.wdg_sel.SpinBoxPtSize.value()
        save_user_settings(
            {
                "pt_size": pt_size,
            },
            mode="update",
        )
        if not hasattr(self, "canvas_points"):
            return None
        self.canvas_points.point_size = pt_size
        self.canvas_points.flush_points_style()

    def reset_points_icon(self):
        """Reset point prompt style"""
        pt_icon_type = self.wdg_sel.comboBoxIconType.currentText()
        save_user_settings(
            {
                "icon_type": pt_icon_type,
            },
            mode="update",
        )
        if not hasattr(self, "canvas_points"):
            return None
        self.canvas_points.icon_type = ICON_TYPE[pt_icon_type]
        self.canvas_points.flush_points_style()

    def reset_rectangular_color(self):
        """Reset rectangular color"""
        color = self.wdg_sel.ColorButton_bbox.color()
        save_user_settings({"bbox_color": color.name()}, mode="update")
        if not hasattr(self, "canvas_rect"):
            return None

        color_fill = list(color.getRgb())
        color_fill[-1] = 10
        color_fill = QColor(*color_fill)
        self.canvas_rect.set_line_color(color)
        self.canvas_rect.set_fill_color(color_fill)

        self.tool_click_rect.reset_cursor_color(color.name())

    def reset_extent_color(self):
        """Reset extent color"""
        color = self.wdg_sel.ColorButton_extent.color()
        if not hasattr(self, "canvas_extent"):
            return None
        self.canvas_extent.set_color(color)
        save_user_settings({"extent_color": color.name()}, mode="update")

        if hasattr(
            self, "sam_extent_canvas_crs"
        ):
            self.canvas_extent.clear()
            self.canvas_extent.add_extent(self.sam_extent_canvas_crs)

    def reset_prompt_polygon_color(self):
        """Reset prompt polygon color"""
        if not hasattr(self, "polygon"):
            return None
        color = self.wdg_sel.ColorButton_prompt.color()
        save_user_settings({"prompt_color": color.name()}, mode="update")
        self.polygon.canvas_prompt_polygon.set_line_style(color)

    def reset_preview_polygon_color(self):
        """Reset preview polygon color"""
        if not hasattr(self, "polygon"):
            return None
        color = self.wdg_sel.ColorButton_preview.color()
        save_user_settings({"preview_color": color.name()}, mode="update")

        self.polygon.canvas_preview_polygon.set_line_style(color)

    def reset_all_styles(self):
        self.reset_points_bg()
        self.reset_points_fg()
        self.reset_points_icon()
        self.reset_points_size()
        self.reset_rectangular_color()
        self.reset_extent_color()
        self.reset_prompt_polygon_color()
        self.reset_preview_polygon_color()