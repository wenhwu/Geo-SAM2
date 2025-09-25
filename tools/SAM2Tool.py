import os
import time
from typing import TYPE_CHECKING
import rasterio
import pandas as pd

import numpy as np
import torch

from qgis.core import QgsCoordinateReferenceSystem, QgsRectangle, QgsProject 

from rasterio.features import shapes as get_shapes
from rasterio.transform import from_bounds as transform_from_bounds 

from .geoTool import LayerExtent, BoundingBox
from .messageTool import MessageTool
from .sam2_ext import Sam2PredictorNoImgEncoder, build_sam2_no_encoder

if TYPE_CHECKING:
    from .widgetTool import Selector

# enable GDAL memory cache for gdal >= 3.10
os.environ["GDAL_MEM_ENABLE_OPEN"] = "YES"


class SAM2_Model:
    def __init__(self, feature_dir, cwd):
        self.feature_dir = feature_dir
        self.sam2_checkpoint = {
            "hiera_l": cwd
            + "/checkpoints/sam2.1_hiera_large.pt",  # hiera large model
            "hiera_b": cwd
            + "/checkpoints/sam2.1_hiera_base_plus.pt",  # hiera base plus model
            "hiera_s": cwd
            + "/checkpoints/sam2.1_hiera_small.pt",  # hiera small model
            "hiera_t": cwd
            + "/checkpoints/sam2.1_hiera_tiny.pt",  # hiera tiny model
        }
        self.sam2_config = {
            "hiera_l": cwd + "/checkpoints/sam2.1_no_encoder/sam2.1_hiera_l.yaml",  # hiera large model
            "hiera_t": cwd + "/checkpoints/sam2.1_no_encoder/sam2.1_hiera_t.yaml",  # hiera tiny model
        }
        
        self.model_type = None
        self.img_crs = None
        self.extent = None
        # self.sample_path = None  # necessary
        self._prepare_data_and_layer()

    def _prepare_data_and_layer(self):
        """Prepares data and layer."""
        self.sam2_feature = {}
        self.image_embed_dir = ""
        self.low_res_feat_dir = ""
        self.high_res_feats_dir_list = []
        self.box_predictor_feats_dir_list = []
        for folder in os.listdir(self.feature_dir):
            if "image_embed" in folder:
                self.image_embed_dir = os.path.join(self.feature_dir, folder)
            elif "high_res_feats" in folder:
                self.high_res_feats_dir_list.append(os.path.join(self.feature_dir, folder))
            elif "box_predictor_feats" in folder:
                self.box_predictor_feats_dir_list.append(os.path.join(self.feature_dir, folder))   
            elif "low_res_feat" in folder:
                self.low_res_feat_dir = os.path.join(self.feature_dir, folder)
        assert self.image_embed_dir, "No image_embed directory found in feature_dir"
        
        # Find and read CSV file
        csv_files = [f for f in os.listdir(self.image_embed_dir) if f.endswith('.csv')]
        if len(csv_files) != 1:
            raise Exception(f"Expected exactly one CSV file in {self.image_embed_dir}, found {len(csv_files)}")
        
        self.image_embed_csv_path = os.path.join(self.image_embed_dir, csv_files[0])
        self.image_embed_df = pd.read_csv(self.image_embed_csv_path)
        self.feature_size = len(self.image_embed_df)
        
        # Ensure required columns exist
        required_columns = ['patch_id', 'filepath', 'model_type', 'feature_type', 'minx', 'miny', 'maxx', 'maxy']
        for col in required_columns:
            if col not in self.image_embed_df.columns:
                raise Exception(f"Required column '{col}' not found in CSV file")
        
        # Set model type
        model_types = self.image_embed_df['model_type'].unique()
        if len(model_types) != 1:
            raise Exception(f"Multiple model types found in CSV: {model_types}")
        self.model_type = model_types[0]
        
        # Set CRS and resolution
        self.img_crs = str(self.image_embed_df['crs'].iloc[0])
        self.img_qgs_crs = QgsCoordinateReferenceSystem(self.img_crs)
        self.res = float(self.image_embed_df['res'].iloc[0])
        
        # Calculate overall extent
        minx = self.image_embed_df['minx'].min()
        maxx = self.image_embed_df['maxx'].max()
        miny = self.image_embed_df['miny'].min()
        maxy = self.image_embed_df['maxy'].max()
        self.extent = QgsRectangle(minx, miny, maxx, maxy)
        
        # Load model
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            raise RuntimeError("CUDA is not available, please check your environment.")

        sam2_model = build_sam2_no_encoder(config_file=self.sam2_config[self.model_type], ckpt_path=self.sam2_checkpoint[self.model_type], device=self.device)
        
        self.predictor = Sam2PredictorNoImgEncoder(sam2_model)
        self.raw_box_results = []  # Store raw results before NMS filtering
        
    def _find_patch(self, extent_union):
        """
        Find corresponding patch based on extent_union
        
        Rules:
        1. If extent_union is completely contained within a patch, select that patch
        2. If extent_union intersects with a patch but is not contained, record it
        3. If two patches intersect with extent_union, immediately report an error
        
        Returns:
            tuple: (patch_info, error_message) - if patch is found successfully, error_message is None
        """
        min_x, max_x, min_y, max_y = extent_union
        contained_patch = None
        intersecting_patch = None
        count_intersection = 0
        for _, row in self.image_embed_df.iterrows():
            patch_minx, patch_miny, patch_maxx, patch_maxy = row['minx'], row['miny'], row['maxx'], row['maxy']
            
            # Check if completely contained
            if (min_x >= patch_minx and max_x <= patch_maxx and 
                min_y >= patch_miny and max_y <= patch_maxy):
                contained_patch = row
                return contained_patch, None
            
            # Check if intersecting
            elif not (max_x < patch_minx or min_x > patch_maxx or 
                     max_y < patch_miny or min_y > patch_maxy):
                count_intersection += 1
                if count_intersection > 1:
                    error_msg = "All prompt points or boxes must be located within the same image patch"
                    return None, error_msg
                intersecting_patch = row
                
        if count_intersection == 1:
            # Only one patch intersects with extent_union
            return intersecting_patch, None
        else:
            # No intersecting patch found
            error_msg = "No image patch found that intersects with the prompt points or boxes"
            return None, error_msg
        
    def _load_patch_features(self, patch_info):
        """Load features for specified patch"""
        patch_id = patch_info['patch_id']
        filepath = os.path.join(self.image_embed_dir, patch_info['filepath'])
        
        # Load image_embed features
        with rasterio.open(filepath) as src:
            image_embed_data = src.read()
            tags = src.tags()
        
        # Convert data type
        # if image_embed_data.dtype == np.uint16:
        #     image_embed_data = image_embed_data.astype(np.int32)
        # elif image_embed_data.dtype == np.uint32:
        #     image_embed_data = image_embed_data.astype(np.int64)
        
        image_embed_tensor = torch.tensor(image_embed_data).float().unsqueeze(0)
        
        # Load high_res_feats
        high_res_feats = []
        for high_res_feat_dir in self.high_res_feats_dir_list:
            high_res_filepath = os.path.join(high_res_feat_dir, str(patch_id) + ".tif")
            if os.path.exists(high_res_filepath) is False:
                raise FileNotFoundError(f"High-res feature file not found: {high_res_filepath}")
            with rasterio.open(high_res_filepath) as src:
                high_res_data = src.read()
            
            high_res_tensor = torch.tensor(high_res_data).float().unsqueeze(0)
            high_res_feats.append(high_res_tensor)
        
        return {
            'image_embed': image_embed_tensor,
            'high_res_feats': high_res_feats,
            'tags': tags,
            'bbox': BoundingBox(patch_info['minx'], patch_info['maxx'], 
                              patch_info['miny'], patch_info['maxy'], patch_info['mint'], patch_info['maxt'])
        }

    def sam2_predict(self, selector: "Selector") -> bool:
        extent_union = LayerExtent.union_extent(
            selector.canvas_points.extent, selector.canvas_rect.extent
        )

        if extent_union is None:
            return True  # no extent to predict

        # Find the correct patch
        patch_info, error_msg = self._find_patch(extent_union)
        if error_msg:
            if selector.preview_mode:
                MessageTool.MessageLog(error_msg, "warning", notify_user=False)
                return True
            else:
                MessageTool.MessageLog(error_msg, "warning")
                return False

        start_time = time.time()
        
        # Check if it's the same patch
        current_patch_id = patch_info['patch_id']
        patch_data = self._load_patch_features(patch_info)
        
        # Set predictor features
        img_height, img_width = 1024, 1024  # default size
        
        self.predictor.set_image_feature(
            sam2_features={
                "image_embed": patch_data['image_embed'],
                "high_res_feats": patch_data['high_res_feats']
            },
            img_size=[(img_height, img_width)],
        )
        
        # Calculate transform
        bbox = patch_data['bbox']
        self.img_clip_transform = transform_from_bounds(
            bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, img_width, img_height
        )
        img_clip_transform = self.img_clip_transform
        
        # self.sample_path = patch_info['filepath']
        MessageTool.MessageLog(f"Load patch {current_patch_id} feature")

        input_point, input_label = selector.canvas_points.get_points_and_labels(
            img_clip_transform
        )

        input_box = selector.canvas_rect.get_img_box(img_clip_transform)
        
        MessageTool.MessageLog(f"input_box   = {input_box}")
        MessageTool.MessageLog(f"input_point = {input_point}")
        
        masks, _, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False,
        )
        
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        MessageTool.MessageLog(f"SAM2 predict executed with {elapsed_time:.3f} ms")

        # Process prediction results
        mask = masks[0, ...].astype(np.uint8)

        shape_generator = get_shapes(
            mask,
            mask=mask,
            connectivity=4,
            transform=img_clip_transform,
        )
        
        geojson = [
            {"properties": {"patch_id": current_patch_id, "score": 1.0}, "geometry": polygon}
            for polygon, value in shape_generator
        ]

        layers = QgsProject.instance().mapLayersByName("polygon_sam2_mask")
        if not layers:   
            MessageTool.MessageLog("Please set a polygon_sam2_mask layer first.")  
            return False
        layer = layers[0]
        if layer is None:
            MessageTool.MessageLog("Please set a polygon_sam2_mask layer first.")
            return False

        selector.polygon.layer = layer
        
        selector.polygon.canvas_preview_polygon.clear()

        target = "prompt"
        if selector.preview_mode:
            target = "preview"

        # Add results to canvas
        selector.polygon.add_geojson_feature_to_canvas(
            geojson,
            selector,
            target=target,
            overwrite_geojson=True,
        )
        return True