import torch
import numpy as np
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List
import json
from datetime import datetime
import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal
import cv2
import torch
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBand,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterCrs,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterEnum,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRange,
    QgsProcessingParameterRasterLayer,
    QgsRasterBandStats,
    QgsRectangle,
    QgsProject,
    QgsUnitTypes,
)
from qgis.gui import QgsDockWidget, QgsFileWidget
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtWidgets import QAction, QDockWidget
from qgis.utils import iface
from torch import Tensor
from .geoTool import BoundingBox

from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms
from sam2.modeling.sam2_base import SAM2Base

# from ..docs import encoder_help
from ..ui.icons import QIcon_EncoderTool
from .messageTool import MessageTool

# 0 for meters, 6 for degrees, 9 for unknown
UNIT_METERS = 0
UNIT_DEGREES = 6


def forward_batch_ui(self, img_list):
    img_batch = [self.transforms(img) for img in img_list]
    img_batch = torch.stack(img_batch, dim=0)
    return img_batch

# Add method to class dynamically
SAM2Transforms.forward_batch_ui = forward_batch_ui


def _prepare_backbone_features_sam2(self, backbone_out):
    """Prepare and flatten visual features."""
    backbone_out = backbone_out.copy()
    assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
    assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

    feature_maps = backbone_out["backbone_fpn"]
    vision_pos_embeds = backbone_out["vision_pos_enc"]
    # feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :] # self.num_feature_levels = 3
    # vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

    feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
    # flatten NxCxHxW to HWxNxC
    vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
    vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

    return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

SAM2Base._prepare_backbone_features_sam2 = _prepare_backbone_features_sam2


class Sam2ProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    This algorithm takes a raster layer and extracts image features using SAM2 encoder.
    It supports processing large images by dividing them into patches.
    """

    # Constants used to refer to parameters and outputs
    INPUT = "INPUT"
    CKPT = "CKPT"
    MODEL_TYPE = "MODEL_TYPE"
    BANDS = "BANDS"
    STRIDE = "STRIDE"
    EXTENT = "EXTENT"
    LOAD = "LOAD"
    OUTPUT = "OUTPUT"
    RANGE = "RANGE"
    RESOLUTION = "RESOLUTION"
    CRS = "CRS"
    CUDA = "CUDA"
    BATCH_SIZE = "BATCH_SIZE"
    CUDA_ID = "CUDA_ID"

    def initAlgorithm(self, config=None):
        """
        Define the inputs and output of the algorithm
        """

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                name=self.INPUT,
                description=self.tr("Input raster layer or image file path"),
            )
        )

        self.addParameter(
            QgsProcessingParameterBand(
                name=self.BANDS,
                description=self.tr(
                    "Select no more than 3 bands (preferably in RGB order, default to first 3 available bands)"
                ),
                parentLayerParameterName=self.INPUT,
                optional=True,
                allowMultiple=True,
            )
        )

        crs_param = QgsProcessingParameterCrs(
            name=self.CRS,
            description=self.tr("Target CRS (default to original CRS)"),
            optional=True,
        )

        res_param = QgsProcessingParameterNumber(
            name=self.RESOLUTION,
            description=self.tr(
                "Target resolution in meters (default to native resolution)"
            ),
            type=QgsProcessingParameterNumber.Double,
            optional=True,
            minValue=0,
            maxValue=100000,
        )

        # expression for scaling the raster values to [0,255]
        range_param = QgsProcessingParameterRange(
            name=self.RANGE,
            description=self.tr(
                "Data value range to be rescaled to [0, 255] (default to [min, max] of the values)"
            ),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=None,
            optional=True,
        )

        cuda_id_param = QgsProcessingParameterNumber(
            name=self.CUDA_ID,
            description=self.tr(
                "CUDA Device ID (choose which GPU to use, default to device 0)"
            ),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=0,
            minValue=0,
            maxValue=9,
        )

        self.addParameter(
            QgsProcessingParameterExtent(
                name=self.EXTENT,
                description=self.tr("Processing extent (default to the entire image)"),
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.STRIDE,
                description=self.tr(
                    "Stride (large image will be sampled into overlapped patches)"
                ),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1024,
                minValue=1024,
                maxValue=1024,
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                name=self.CKPT,
                description=self.tr("SAM2 checkpoint path (download in advance)"),
                extension="pt",
            )
        )

        self.model_type_options = ["hiera_l", "hiera_t"]
        self.addParameter(
            QgsProcessingParameterEnum(
                name=self.MODEL_TYPE,
                description=self.tr("SAM2 model type (l for large, t for tiny)"),
                options=self.model_type_options,
                defaultValue=0,  # 'hiera_l'
            )
        )

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT,
                self.tr("Output directory (choose the location to save image features)"),
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CUDA, self.tr("Use GPU if CUDA is available."), defaultValue=True
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.BATCH_SIZE,
                description=self.tr(
                    "Batch size (take effect if choose to use GPU and CUDA is available)"
                ),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1,
                minValue=1,
                maxValue=1,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.LOAD,
                self.tr("Load output features in Geo-SAM2 tool after processing"),
                defaultValue=True,
            )
        )

        for param in (crs_param, res_param, range_param, cuda_id_param):
            param.setFlags(
                param.flags() | QgsProcessingParameterDefinition.FlagAdvanced
            )
            self.addParameter(param)

        # Spatial dimensions for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        
        mask_threshold=0.0
        max_hole_area=0.0
        max_sprinkle_area=0.0
        image_size = 1024
        self._transforms = SAM2Transforms(
            resolution=image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

    def _gdal_read_patch(self, poRasterDS, adjusted_topLeftX, adjusted_topLeftY, nPatchCols, nPatchRows, feedback):
        """
        Use GDAL to read specified image patch with complete dimensions.
        
        Different from original version, this version automatically adjusts the starting point 
        when the crop area is close to the right or bottom edge to ensure reading a complete 
        (nPatchCols, nPatchRows) sized image patch without relying on mirror padding.
        
        Args:
            poRasterDS: GDAL dataset object.
            topLeftX (int): Desired top-left X coordinate of crop area.
            topLeftY (int): Desired top-left Y coordinate of crop area.
            nPatchCols (int): Target image patch width.
            nPatchRows (int): Target image patch height.

        Returns:
            numpy.ndarray: Read and merged image patch (H, W, C), size (nPatchRows, nPatchCols, 3).
                        Returns None if error occurs.
        """

        # SAM model requires 3-channel input, adapt based on user-selected bands
        if len(self.selected_bands) == 1:
            band_indices_to_read = self.selected_bands * 3
        elif len(self.selected_bands) == 2:
            band_indices_to_read = self.selected_bands + [self.selected_bands[0]]
        else:
            band_indices_to_read = self.selected_bands[:3]

        vImgMat = []
        for band_index in band_indices_to_read:
            pBand = poRasterDS.GetRasterBand(band_index)
            if pBand is None:
                feedback.pushInfo(f"Error: Unable to get band {band_index}.")
                return None
            
            # Use adjusted coordinates for reading
            pafscan = pBand.ReadAsArray(adjusted_topLeftX, adjusted_topLeftY, nPatchCols, nPatchRows)
            
            if pafscan is None or pafscan.size == 0:
                # Under normal circumstances, we shouldn't get empty data here due to coordinate adjustment
                feedback.pushInfo(f"Warning: Reading band {band_index} at adjusted coordinates ({adjusted_topLeftX}, {adjusted_topLeftY}) returned empty data.")
                return None
            
            vImgMat.append(pafscan)

        # Check if all channel shapes are consistent
        if not all(channel.shape == vImgMat[0].shape for channel in vImgMat):
            feedback.pushInfo("Error: Input channel shapes do not match.")
            return None
        
        # Merge channels (H, W, C)
        img_patch = cv2.merge(vImgMat)

        return img_patch


    def processAlgorithm(self, parameters, context, feedback):
        """
        Main processing logic
        """

        alg_start_time = time.time()
        dt = datetime.fromtimestamp(alg_start_time)
        feedback.pushInfo(f"Algorithm start time = {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.iPatch = 0
        self.feature_dir = ""

        rlayer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if rlayer is None:
            raise QgsProcessingException(
                self.invalidRasterError(parameters, self.INPUT)
            )
        

        self.selected_bands = self.parameterAsInts(parameters, self.BANDS, context)

        if len(self.selected_bands) == 0:
            max_band = min(3, rlayer.bandCount())
            self.selected_bands = list(range(1, max_band + 1))

        if len(self.selected_bands) > 3:
            raise QgsProcessingException(
                self.tr("Please choose no more than three bands!")
            )
        if max(self.selected_bands) > rlayer.bandCount():
            raise QgsProcessingException(
                self.tr("The chosen bands exceed the largest band number!")
            )

        ckpt_path = self.parameterAsFile(parameters, self.CKPT, context)
        model_type_idx = self.parameterAsEnum(parameters, self.MODEL_TYPE, context)
        stride = self.parameterAsInt(parameters, self.STRIDE, context)
        res = self.parameterAsDouble(parameters, self.RESOLUTION, context)
        crs = self.parameterAsCrs(parameters, self.CRS, context)
        extent = self.parameterAsExtent(parameters, self.EXTENT, context)
        self.load_feature = self.parameterAsBoolean(parameters, self.LOAD, context)
        self.use_gpu = self.parameterAsBoolean(parameters, self.CUDA, context)
        batch_size = self.parameterAsInt(parameters, self.BATCH_SIZE, context)
        range_value = self.parameterAsRange(parameters, self.RANGE, context)
        output_dir = self.parameterAsString(parameters, self.OUTPUT, context)
        self.cuda_id = self.parameterAsInt(parameters, self.CUDA_ID, context)
    
        rlayer_data_provider = rlayer.dataProvider()

        out_path = os.path.join(output_dir, rlayer.name())
        # remove existing output folder to avoid mixing old and new results
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
            
        # handle crs
        if crs is None or not crs.isValid():
            crs = rlayer.crs()

        if rlayer.crs().mapUnits() == UNIT_DEGREES:
            layer_units = "degrees"
        else:
            layer_units = "meters"
        # if res is not provided, get res info from rlayer
        if np.isnan(res) or res == 0:
            res = rlayer.rasterUnitsPerPixelX()
            target_units = layer_units
        else:
            # when given res in meters by users, convert crs to utm if the original crs unit is degree
            if crs.mapUnits() != UNIT_METERS:
                if rlayer.crs().mapUnits() == UNIT_DEGREES:
                    # estimate utm crs based on layer extent
                    crs = self.estimate_utm_crs(rlayer.extent())
                else:
                    raise QgsProcessingException(
                        f"Resampling of image with the CRS of {crs.authid()} in meters is not supported."
                    )
            target_units = "meters"

        self.res = res

        # handle extent
        if extent.isNull():
            extent = rlayer.extent()
            extent_crs = rlayer.crs()
        else:
            if extent.isEmpty():
                raise QgsProcessingException(
                    self.tr("The extent for processing cannot be empty!")
                )
            
            extent_crs = self.parameterAsExtentCrs(parameters, self.EXTENT, context)

        # if extent crs != target crs, convert it to target crs
        if extent_crs != crs:
            transform = QgsCoordinateTransform(
                extent_crs, crs, context.transformContext()
            )
            extent_polygon = QgsGeometry.fromRect(extent)
            extent_polygon.transform(transform)
            extent = extent_polygon.boundingBox()
            extent_crs = crs

        # check intersects between extent and rlayer_extent
        if rlayer.crs() != crs:
            transform = QgsCoordinateTransform(
                rlayer.crs(), crs, context.transformContext()
            )
            rlayer_extent = transform.transformBoundingBox(rlayer.extent())
        else:
            rlayer_extent = rlayer.extent()
        if not rlayer_extent.intersects(extent):
            raise QgsProcessingException(
                self.tr("The extent for processing is not intersected with the input image!")
            )

        model_type = self.model_type_options[model_type_idx]
        if model_type not in os.path.basename(ckpt_path):
            raise QgsProcessingException(
                self.tr("Model type does not match the checkpoint")
            )
            
        img_width_in_extent = round((extent.xMaximum() - extent.xMinimum()) / self.res)
        img_height_in_extent = round((extent.yMaximum() - extent.yMinimum()) / self.res)
        
        # handle value range
        if (not np.isnan(range_value[0])) and (not np.isnan(range_value[1])):
            feedback.pushInfo(
                f"Input data value range to be rescaled: {range_value} (set by user)"
            )
        else:
            if extent_crs == rlayer.crs():
                stat_extent = extent
            else:
                transform = QgsCoordinateTransform(
                    extent_crs, rlayer.crs(), context.transformContext()
                )
                stat_extent = transform.transformBoundingBox(extent)
            start_time = time.time()
            # set sample size to limit statistic time
            sample_size = min(int(1e8), img_height_in_extent * img_width_in_extent)
            min_values = []
            max_values = []
            for band in self.selected_bands:
                band_stats = rlayer_data_provider.bandStatistics(
                    bandNo=band,
                    stats=QgsRasterBandStats.All,
                    extent=stat_extent,
                    sampleSize=sample_size,
                )
                min_values.append(band_stats.minimumValue)
                max_values.append(band_stats.maximumValue)
            range_value[0] = min(min_values)
            range_value[1] = max(max_values)
            end_time = time.time()
            elapsed_time = end_time - start_time
            feedback.pushInfo(
                f"Input data value range to be rescaled: {range_value} (automatically set based on min-max value of input image inside the processing extent.)"
            )
            feedback.pushInfo(f"Band statistics took time {elapsed_time:.3f}s")

        if range_value[0] >= range_value[1]:
            raise QgsProcessingException(
                self.tr("Data value range is wrongly set or the image has constant values.")
            )

        # Send some information to the user
        feedback.pushInfo(f"Layer path: {rlayer_data_provider.dataSourceUri()}")
        feedback.pushInfo(f"Layer name: {rlayer.name()}")
        if rlayer.crs().authid():
            feedback.pushInfo(f"Layer CRS: {rlayer.crs().authid()}")
        else:
            feedback.pushInfo(f"Layer CRS in WKT format: {rlayer.crs().toWkt()}")
        feedback.pushInfo(
            f"Layer pixel size: {rlayer.rasterUnitsPerPixelX()}, {rlayer.rasterUnitsPerPixelY()} {layer_units}"
        )

        feedback.pushInfo(f"Bands selected: {self.selected_bands}")

        if crs.authid():
            feedback.pushInfo(f"Target CRS: {crs.authid()}")
        else:
            feedback.pushInfo(f"Target CRS in WKT format: {crs.toWkt()}")

        feedback.pushInfo(f"Target resolution: {self.res} {target_units}")
        feedback.pushInfo(
            (
                f"Processing extent: minx:{extent.xMinimum():.6f}, maxx:{extent.xMaximum():.6f},"
                f"miny:{extent.yMinimum():.6f}, maxy:{extent.yMaximum():.6f}"
            )
        )
        feedback.pushInfo(
            (
                f"Processing image size: (width {img_width_in_extent}, "
                f"height {img_height_in_extent})"
            )
        )

        # Get layer path
        rlayer_path = rlayer.source()

        all_bands = [
            rlayer.bandName(i_band) for i_band in range(1, rlayer.bandCount() + 1)
        ]
        # currently only support rgb bands
        input_bands = [rlayer.bandName(i_band) for i_band in self.selected_bands]
        # ensure only three bands are used, less than three bands will be broadcasted to three bands
        input_bands = (input_bands * 3)[0:3]

        feedback.pushInfo(
            f"\n RasterDataset info: \
            \n filename: {rlayer_path}, \
            \n all bands: {all_bands}, \
            \n input bands: {input_bands}, \
            \n resolution: {res}, \
            \n bounds: {extent}\n"
        )

        self.sam2_model = self.initialize_sam2(
            model_type=model_type, sam_ckpt_path=ckpt_path
        )
        patch_size = self.sam2_model.image_size

        # --- GDAL sliding window setup ---
        gdal.AllRegister()
        rlayer_path = rlayer.dataProvider().dataSourceUri()
        poRasterDS = gdal.Open(rlayer_path, gdal.GA_ReadOnly)
        if poRasterDS is None:
            raise Exception(f"Failed to load image using GDAL: {rlayer_path}")

        nCols = poRasterDS.RasterXSize
        nRows = poRasterDS.RasterYSize
        adfGeoTransform = poRasterDS.GetGeoTransform()
        strProjection = poRasterDS.GetProjection()

        feedback.pushInfo(f"Image dimensions: (width={nCols}, height={nRows})")
        feedback.pushInfo("--------------------------------------------------------------")

        # --- New: Determine if patch processing is needed ---
        if nCols <= 1024 and nRows <= 1024:
            # Image dimensions do not exceed 1024×1024, process entire image directly
            feedback.pushInfo("Image dimensions do not exceed 1024×1024, processing entire image directly")
            
            # Read entire image
            img_patch_np = self._gdal_read_patch(
                poRasterDS, 0, 0, nCols, nRows, feedback=feedback
            )
            
            if img_patch_np is None:
                feedback.pushInfo("Failed to read entire image")
                return {"Output feature_dir path": "", "Patch samples saved": 0, "Feature folder loaded": False}
            
            if np.all(img_patch_np == 0):
                feedback.pushInfo("Entire image is all zeros, skipping processing")
                return {"Output feature_dir path": "", "Patch samples saved": 0, "Feature folder loaded": False}
            
            # Process entire image
            img_patch_chw = img_patch_np.transpose((2, 0, 1))
            batch_tensor = torch.from_numpy(img_patch_chw).unsqueeze(0).float()
            
            self.batch_input = self.rescale_img(batch_tensor, range_value)
            
            feedback.pushInfo("Processing entire image...")
            feedback.pushInfo(f"Input image shape = {self.batch_input.shape}")
            
            batch_input_norm = self.batch_input.div(255.0)
            img_batch = self._transforms.forward_batch_ui([img for img in batch_input_norm])
            self.batch_input = img_batch.to(self.sam2_model.device)
            
            if not self.get_sam2_feature(self.batch_input, feedback=feedback):
                self.load_feature = False
                return {"Output feature_dir path": "", "Patch samples saved": 0, "Feature folder loaded": False}
            else:
                self.load_feature = True
            
            # Save features
            minx = adfGeoTransform[0]
            maxy = adfGeoTransform[3]
            maxx = minx + nCols * adfGeoTransform[1]
            miny = maxy + nRows * adfGeoTransform[5]
            bbox = BoundingBox(minx, maxx, miny, maxy, mint=0, maxt=1)
            
            mock_batch = {
                'path': [rlayer_path],
                'bbox': [bbox],
                'crs': [strProjection],
                'img_shape': [(nRows, nCols)],
                'input_shape': [self.batch_input.shape[2:]]
            }
            
            # Save various features
            patch_id_for_save = 0  # Entire image as one patch, ID 0
            self.image_embed_feature_dir = self.save_sam2_feature(
                output_dir, mock_batch, self._features["image_embed"], patch_id_for_save, model_type, "image_embed"
            )
            self.high_res_feats_dir = [
                self.save_sam2_feature(
                    output_dir, mock_batch, feature, patch_id_for_save, model_type, "high_res_feats_" + str(i)
                ) for i, feature in enumerate(self._features["high_res_feats"])
            ]
            
            self.iPatch = 1  # Processed 1 patch (entire image)
            
        else:
            # Image dimensions exceed 1024×1024, perform patch processing (original code)
            feedback.pushInfo("Image dimensions exceed 1024×1024, performing patch processing")
            
            # Calculate number of patches (no overlap)
            quotientCols = nCols // patch_size
            nResidueCols = nCols % patch_size
            numCols = quotientCols + (1 if nResidueCols > 0 else 0)

            quotientRows = nRows // patch_size
            nResidueRows = nRows % patch_size
            numRows = quotientRows + (1 if nResidueRows > 0 else 0)
            
            total_patches = numCols * numRows
            feedback.pushInfo(f"Total number of patches to process: {total_patches}")
            
            elapsed_time_list = []
            for j in range(numRows):
                for i in range(numCols):
                    if feedback.isCanceled():
                        self.load_feature = False
                        feedback.pushWarning(
                            self.tr("\n !!!Processing canceled by user!!! \n")
                        )
                        break
                    current_patch_idx = j * numCols + i
                    start_time = time.time()
                    
                    # Calculate current patch pixel coordinates and dimensions
                    oriTopLeftX = i * patch_size
                    oriTopLeftY = j * patch_size
                    
                    nPatchCols = patch_size 
                    nPatchRows = patch_size

                    nImgCols = poRasterDS.RasterXSize
                    nImgRows = poRasterDS.RasterYSize

                    # Check if crop area exceeds right boundary
                    if oriTopLeftX + nPatchCols > nImgCols:
                        # If exceeds, move starting X coordinate left to align right edge with image right edge
                        adjusted_topLeftX = nImgCols - nPatchCols
                    else:
                        adjusted_topLeftX = oriTopLeftX

                    # Check if crop area exceeds bottom boundary
                    if oriTopLeftY + nPatchRows > nImgRows:
                        # If exceeds, move starting Y coordinate up to align bottom edge with image bottom edge
                        adjusted_topLeftY = nImgRows - nPatchRows
                    else:
                        adjusted_topLeftY = oriTopLeftY
                        
                    # Ensure coordinates are not negative (when image is smaller than target size)
                    adjusted_topLeftX = max(0, adjusted_topLeftX)
                    adjusted_topLeftY = max(0, adjusted_topLeftY)

                    img_patch_np = self._gdal_read_patch(
                        poRasterDS, int(adjusted_topLeftX), int(adjusted_topLeftY), int(nPatchCols), int(nPatchRows), feedback = feedback
                    )

                    if img_patch_np is None:
                        feedback.pushInfo(f"Skipping empty or invalid patch at ({adjusted_topLeftX}, {adjusted_topLeftY})")
                        continue
                    
                    if np.all(img_patch_np == 0):
                        feedback.pushInfo(f"Skipping all-zero patch at ({adjusted_topLeftX}, {adjusted_topLeftY})")
                        continue

                    # --- Model inference ---
                    # Convert numpy HWC to torch tensor BCHW
                    img_patch_chw = img_patch_np.transpose((2, 0, 1))
                    batch_tensor = torch.from_numpy(img_patch_chw).unsqueeze(0).float()

                    self.batch_input = self.rescale_img(batch_tensor, range_value)
                    
                    feedback.pushInfo(f"Processing Patch {current_patch_idx + 1}/{total_patches}...")
                    feedback.pushInfo(f"Input Patch shape = {self.batch_input.shape}")
                    
                    batch_input_norm = self.batch_input.div(255.0)
                    img_batch = self._transforms.forward_batch_ui([img for img in batch_input_norm])
                    self.batch_input = img_batch.to(self.sam2_model.device)
                    
                    if not self.get_sam2_feature(self.batch_input, feedback=feedback):
                        self.load_feature = False
                        break
                    
                    # --- Save features ---
                    # Manually calculate geographic bounding box of current patch
                    minx = adfGeoTransform[0] + adjusted_topLeftX * adfGeoTransform[1]
                    maxy = adfGeoTransform[3] + adjusted_topLeftY * adfGeoTransform[5]
                    maxx = minx + nPatchCols * adfGeoTransform[1]
                    miny = maxy + nPatchRows * adfGeoTransform[5]
                    bbox = BoundingBox(minx, maxx, miny, maxy, mint=0, maxt=1)

                    # Build mock 'batch' dictionary for save function
                    mock_batch = {
                        'path': [rlayer_path],
                        'bbox': [bbox],
                        'crs': [strProjection],
                        'img_shape': [(nRows, nCols)],
                        'input_shape': [self.batch_input.shape[2:]]
                    }
                    
                    # Save various features
                    patch_id_for_save = current_patch_idx
                    self.image_embed_feature_dir = self.save_sam2_feature(
                        output_dir, mock_batch, self._features["image_embed"], patch_id_for_save, model_type, "image_embed"
                    )
                    self.high_res_feats_dir = [
                        self.save_sam2_feature(
                            output_dir, mock_batch, feature, patch_id_for_save, model_type,  "high_res_feats_" + str(i)
                        ) for i, feature in enumerate(self._features["high_res_feats"])
                    ]
                
                    self.iPatch += 1

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    elapsed_time_list.append(elapsed_time)
                    time_spent = sum(elapsed_time_list)
                    time_remain = (time_spent / (current_patch_idx + 1)) * (total_patches - current_patch_idx - 1)
                    
                    feedback.pushInfo(f"SAM2 encoder execution time {elapsed_time:.3f}s")
                    if time_remain <= 60:
                        feedback.pushInfo(f"Estimated remaining time: {time_remain:.3f}s")
                    else:
                        time_remain_m, time_remain_s = divmod(int(time_remain), 60)
                        time_remain_h, time_remain_m = divmod(time_remain_m, 60)
                        feedback.pushInfo(f"Estimated remaining time: {time_remain_h:d}h:{time_remain_m:02d}m:{time_remain_s:02d}s")
                    feedback.pushInfo("--------------------------------------------------------------")
                    feedback.setProgress(int((current_patch_idx + 1) / total_patches*100))

        # --- Cleanup and finalization ---
        poRasterDS = None # Close GDAL dataset
        alg_end_time = time.time()
        feedback.pushInfo(f"Total algorithm execution time: {alg_end_time - alg_start_time:.3f}s")
        
        # Build formatted extent string
        extent_str = (
            f"{extent.xMinimum()},{extent.xMaximum()},"
            f"{extent.yMinimum()},{extent.yMaximum()} "
            f"[{extent_crs.authid()}]"
        )
         
        # Collect all parameter values
        params_dict = {
            "BANDS": self.selected_bands,
            "BATCH_SIZE": batch_size,
            "CKPT": ckpt_path,
            "CRS": crs.authid() if crs and crs.isValid() else "None",
            "CUDA": self.use_gpu,
            "CUDA_ID": self.cuda_id,
            "EXTENT": extent_str,
            "INPUT": rlayer.source() if rlayer else "None",
            "LOAD": self.load_feature,
            "MODEL_TYPE": model_type_idx,
            "OUTPUT": output_dir,
            "RANGE": f"{range_value[0]},{range_value[1]}" if range_value else "None,None",
            "RESOLUTION": res,
            "STRIDE": stride
        }
        
        # Get project settings
        project = QgsProject.instance()
        full_params = {
            "area_units": QgsUnitTypes.toString(project.areaUnits()),
            "distance_units": QgsUnitTypes.toString(project.distanceUnits()),
            "ellipsoid": project.ellipsoid(),
            "timestamp": datetime.now().isoformat(),
            "inputs": params_dict
        }
        
        # Save as JSON file
        out_path = os.path.join(output_dir, rlayer.name())
        MessageTool.MessageLog(f"Saving parameters to {out_path}")
        param_file = os.path.join(output_dir, rlayer.name(), "sam2_encoder_parameters.json")
        with open(param_file, 'w') as f:
            json.dump(full_params, f, indent=4)
        
        feedback.pushInfo(f"\nParameters saved to {param_file}")
        self.feature_dir = out_path

        return {
            "Output feature_dir path": self.feature_dir,
            "Patch samples saved": self.iPatch,
            "Feature folder loaded": self.load_feature,
        }

    # Used to handle any thread-sensitive cleanup required by the algorithm
    def postProcessAlgorithm(self, context, feedback) -> Dict[str, Any]:
        if torch.cuda.is_available() and self.use_gpu:
            if hasattr(self, "sam2_model"):
                del self.sam2_model
            if hasattr(self, "batch_input"):
                del self.batch_input
            torch.cuda.empty_cache()
        if self.load_feature and self.feature_dir:
            self.load_feature = self.load_feature_dir(feedback=feedback)
        return {
            "Output feature path": self.feature_dir,
            "Patch samples saved": self.iPatch,
            "Feature folder loaded": self.load_feature,
        }

    def load_feature_dir(self, feedback: QgsProcessingFeedback) -> bool:
        sam_tool_action: QAction = iface.mainWindow().findChild(
            QAction, "mActionGeoSamTool"
        )
        if sam_tool_action:
            sam_tool_action.trigger()
            start_time = time.time()
            while True:
                if feedback.isCanceled():
                    feedback.pushInfo(self.tr("Loading feature canceled by user."))
                    return False
                sam_tool_widget: QgsDockWidget = iface.mainWindow().findChild(
                    QDockWidget, "Geo_SAM2"
                )
                current_time = time.time()
                elapsed_time = (current_time - start_time) * 1000
                if sam_tool_widget:
                    feedback.pushInfo("\n Geo_SAM2 widget found")
                    load_feature_widget: QgsFileWidget = sam_tool_widget.QgsFile_feature
                    load_feature_widget.setFilePath(self.feature_dir)
                    sam_tool_widget.pushButton_load_feature.click()  # try sender
                    feedback.pushInfo(
                        f"features in {sam_tool_widget.QgsFile_feature.filePath()} loaded in {elapsed_time:.3f} ms \n"
                    )
                    return True
                # try 3 seconds
                if elapsed_time > 3000:
                    feedback.pushInfo(
                        f"\n Geo_SAM2 widget not found {elapsed_time:.3f} ms \n"
                    )
                    return False
        else:
            feedback.pushInfo("\n Geo_SAM2 tool action not found. \n")
            return False
    
    def initialize_sam2(self, model_type: str, sam_ckpt_path: str):
        
        if model_type == "hiera_t":
            model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        elif model_type == "hiera_s":
            model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif model_type == "hiera_l":
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        elif model_type == "hiera_b":  # base_plus (default)
            model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        
        if torch.cuda.is_available() and self.use_gpu:
            device = torch.device("cuda")
            sam2_model = build_sam2(model_cfg, sam_ckpt_path, device=device)
            return sam2_model
        else:
            return None

  
    def to_numpy(self, feat):
        if isinstance(feat, list):
            feat = torch.stack(feat, dim=0)
        return feat.detach().cpu().numpy()
    
    @torch.no_grad()
    def get_sam2_feature(
        self, batch_input: Tensor, feedback: QgsProcessingFeedback
    ) -> bool:

        batch_size = batch_input.shape[0]

        try:
            backbone_out = self.sam2_model.forward_image(batch_input)
            _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
            # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
            if self.sam2_model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
            feats = [
                feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
            ][::-1]
            features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        except RuntimeError as inst:
            # torch.cuda.OutOfMemoryError
            if "CUDA out of memory" in inst.args[0]:
                feedback.pushWarning(
                    "\n !!!CUDA out of memory, try to choose a smaller batch size or smaller version of SAM2 model.!!!"
                )
                feedback.pushWarning(
                    f"Error type: {type(inst).__name__}, context: {inst} \n"
                )
            return False
        except Exception as err:
            raise QgsProcessingException(f"Unexpected {err=}, {type(err)=}")
        
        self._features = {}
        image_embed = self.to_numpy(features["image_embed"]) 
        high_res_feats = []
        for j, feature in enumerate(features["high_res_feats"]):
            if isinstance(feature, Tensor):
                feature = self.to_numpy(feature) 
                high_res_feats.append(feature) 

        self._features["image_embed"] = image_embed 
        self._features["high_res_feats"] = high_res_feats
        
        return True

    def rescale_img(self, batch_input: Tensor, range_value: List[float]) -> Tensor:
        "rescale input image to [0,255]"
        range_min = range_value[0]
        range_max = range_value[1]
        batch_output = (batch_input - range_min) * 255 / (range_max - range_min)
        return batch_output


    def save_sam2_feature(
        self,
        export_dir_str: str,
        data_batch: Tensor,
        feature: np.ndarray,
        patch_id: int,
        model_type: str,
        feature_type: str,
    ) -> str:
        export_dir = Path(export_dir_str)
        bands_str = "_".join([str(band) for band in self.selected_bands])
        # one image file encoding situation
        filepath = Path(data_batch["path"][0])
        export_dir_sub = (
            export_dir
            / filepath.stem
            / f"{model_type}_{feature_type}_bands_{bands_str}"
        )
        export_dir_sub.mkdir(parents=True, exist_ok=True)
        # iterate over batch_size dimension
        band_num = feature.shape[-3]
        height = feature.shape[-2]
        width = feature.shape[-1]
        for idx in range(feature.shape[-4]):
            bbox = data_batch["bbox"][idx]
            rio_transform = rasterio.transform.from_bounds(
                bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, width, height
            )  # west, south, east, north, width, height
            
            feature_tiff = export_dir_sub / f"{patch_id}.tif"
            feature_csv = export_dir_sub / f"{export_dir_sub.name}.csv"

            with rasterio.open(
                feature_tiff,
                mode="w",
                driver="GTiff",
                height=height,
                width=width,
                count=band_num,
                dtype="float32",
                crs=data_batch["crs"][idx],
                transform=rio_transform,
            ) as feature_dataset:
                feature_dataset.write(feature[idx, ...], range(1, band_num + 1))
                tags = {
                    "img_shape": data_batch["img_shape"][idx],
                    "input_shape": data_batch["input_shape"][idx],
                    "model_type": model_type,
                    "feature_type": feature_type,
                }
                feature_dataset.update_tags(**tags)
                feature_crs = feature_dataset.crs

                index_df = pd.DataFrame(
                    {
                        "patch_id":    [patch_id],
                        "filepath":    [feature_tiff.name],
                        "model_type":  [model_type],
                        "feature_type":[feature_type],
                        "crs":         [str(feature_crs)],
                        "res":         [self.res],
                        "minx":        [bbox.minx],
                        "miny":        [bbox.miny],
                        "maxx":        [bbox.maxx],
                        "maxy":        [bbox.maxy],
                        "mint":        [bbox.mint],
                        "maxt":        [bbox.maxt],
                    },
                )

                index_df.to_csv(
                    feature_csv,
                    mode="a",
                    header=not feature_csv.exists(), 
                    index=False
                )

        return str(export_dir_sub)

    def estimate_utm_crs(self, extent: QgsRectangle):
        utm_crs_list = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(
                west_lon_degree=extent.xMinimum(),
                south_lat_degree=extent.yMinimum(),
                east_lon_degree=extent.xMaximum(),
                north_lat_degree=extent.yMaximum(),
            ),
        )
        utm_crs = CRS.from_epsg(utm_crs_list[0].code)
        utm_crs = QgsCoordinateReferenceSystem(str(utm_crs))
        return utm_crs

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        return Sam2ProcessingAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm.
        """
        return "geo_sam2_encoder"

    def displayName(self):
        """
        Returns the translated algorithm name for user display.
        """
        return self.tr("Geo-SAM2 Image Encoder")

    def group(self):
        """
        Returns the name of the group this algorithm belongs to.
        """
        return self.tr("")

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to.
        """
        return ""

    # def shortHelpString(self):
    #     """
    #     Returns a localized short helper string for the algorithm.
    #     """
    #     file = encoder_help
    #     if not os.path.exists(file):
    #         return self.tr("Generate image features using SAM2 image encoder.")
    #     with open(file) as help_file:
    #         help_str = help_file.read()
    #     return help_str

    def icon(self):
        return QIcon_EncoderTool