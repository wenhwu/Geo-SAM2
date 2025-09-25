from __future__ import annotations  
from qgis.core import (QgsProject,
                       QgsCoordinateReferenceSystem,
                       QgsCoordinateTransform,
                       QgsPointXY,
                       QgsRectangle,
                       QgsVectorLayer)
from .messageTool import MessageTool
from typing import overload
from dataclasses import dataclass
from collections.abc import Iterator


class ImageCRSManager:
    '''Manage image crs and transform point and extent between image crs and other crs'''

    def __init__(self, img_crs) -> None:
        self.img_crs = QgsCoordinateReferenceSystem(
            img_crs)  # from str to QgsCRS

    def img_point_to_crs(
        self, point: QgsPointXY, dst_crs: QgsCoordinateReferenceSystem
    ) -> QgsRectangle:
        """transform point from this image crs to destination crs

        Parameters:
        ----------
        point: QgsPointXY
            point in this image crs
        dst_crs: QgsCoordinateReferenceSystem
            destination crs for point
        """
        if dst_crs == self.img_crs:
            return point
        transform = QgsCoordinateTransform(
            self.img_crs, dst_crs, QgsProject.instance())
        point_transformed = transform.transform(point)
        return point_transformed

    def point_to_img_crs(
        self, point: QgsPointXY, dst_crs: QgsCoordinateReferenceSystem
    ) -> QgsRectangle:
        """transform point from point crs to this image crs

        Parameters:
        ----------
        point: QgsPointXY
            point in itself crs
        point_crs: QgsCoordinateReferenceSystem
            crs of point
        """
        if dst_crs == self.img_crs:
            return point
        transform = QgsCoordinateTransform(
            dst_crs, self.img_crs, QgsProject.instance()
        )
        point_transformed = transform.transform(point)  # direction can be used
        return point_transformed

    def extent_to_img_crs(
        self, extent: QgsRectangle, dst_crs: QgsCoordinateReferenceSystem
    ) -> QgsRectangle:
        """transform extent from point crs to this image crs

        Parameters:
        ----------
        extent: QgsRectangle
            extent in itself crs
        dst_crs: QgsCoordinateReferenceSystem
            destination crs for extent
        """
        if dst_crs == self.img_crs:
            return extent
        transform = QgsCoordinateTransform(
            dst_crs, self.img_crs, QgsProject.instance())
        extent_transformed = transform.transformBoundingBox(extent)
        return extent_transformed

    def img_extent_to_crs(
        self, extent: QgsRectangle, dst_crs: QgsCoordinateReferenceSystem
    ) -> QgsRectangle:
        '''transform extent from this image crs to destination crs

        Parameters:
        ----------
        extent: QgsRectangle
            extent in this image crs
        dst_crs: QgsCoordinateReferenceSystem
            destination crs for extent
        '''
        if dst_crs == self.img_crs:
            return extent
        transform = QgsCoordinateTransform(
            self.img_crs, dst_crs, QgsProject.instance())
        extent_transformed = transform.transformBoundingBox(extent)
        return extent_transformed


class LayerExtent:
    def __init__(self):
        pass

    @staticmethod
    def from_qgis_extent(extent: QgsRectangle):
        max_x = extent.xMaximum()
        max_y = extent.yMaximum()
        min_x = extent.xMinimum()
        min_y = extent.yMinimum()
        return min_x, max_x, min_y, max_y

    @classmethod
    def get_layer_extent(
        cls, layer: QgsVectorLayer, img_crs_manager: ImageCRSManager = None
    ):
        """Get the extent of the layer"""
        if layer.featureCount() == 0:
            return None
        else:
            layer.updateExtents()
            layer_ext = layer.extent()
            if layer.crs() != img_crs_manager.img_crs:
                try:
                    layer_ext = img_crs_manager.extent_to_img_crs(
                        layer_ext, layer.crs())
                except Exception as e:
                    MessageTool.MessageLog(
                        f">>> Error in extent: {layer_ext} \n type:{type(layer_ext)} \n: {e}",
                        level='critical',
                        notify_user=False
                    )

                    return None

            return cls.from_qgis_extent(layer_ext)

    @staticmethod
    def _union_extent(extent1, extent2):
        """Get the union of two extents"""
        min_x1, max_x1, min_y1, max_y1 = extent1
        min_x2, max_x2, min_y2, max_y2 = extent2

        min_x = min(min_x1, min_x2)
        max_x = max(max_x1, max_x2)
        min_y = min(min_y1, min_y2)
        max_y = max(max_y1, max_y2)

        return min_x, max_x, min_y, max_y

    @classmethod
    def union_extent(cls, extent1, extent2):
        """Get the union of two extents (None is allowed)"""
        if extent1 is not None and extent2 is not None:
            min_x, max_x, min_y, max_y = cls._union_extent(extent1, extent2)
        elif extent1 is None and extent2 is not None:
            min_x, max_x, min_y, max_y = extent2
        elif extent1 is not None and extent2 is None:
            min_x, max_x, min_y, max_y = extent1
        else:
            return None

        return min_x, max_x, min_y, max_y

    @classmethod
    def union_layer_extent(
        cls, layer1, layer2, img_crs_manager: ImageCRSManager = None
    ):
        """Get the union of two layer extents"""
        extent1 = cls.get_layer_extent(layer1, img_crs_manager)
        extent2 = cls.get_layer_extent(layer2, img_crs_manager)

        return cls.union_extent(extent1, extent2)


@dataclass(frozen=True)
class BoundingBox:
    """Data class for indexing spatiotemporal data."""

    #: western boundary
    minx: float
    #: eastern boundary
    maxx: float
    #: southern boundary
    miny: float
    #: northern boundary
    maxy: float
    #: earliest boundary
    mint: float
    #: latest boundary
    maxt: float

    def __post_init__(self) -> None:
        """Validate the arguments passed to :meth:`__init__`.

        Raises:
            ValueError: if bounding box is invalid
                (minx > maxx, miny > maxy, or mint > maxt)

        .. versionadded:: 0.2
        """
        if self.minx > self.maxx:
            raise ValueError(
                f"Bounding box is invalid: 'minx={self.minx}' > 'maxx={self.maxx}'"
            )
        if self.miny > self.maxy:
            raise ValueError(
                f"Bounding box is invalid: 'miny={self.miny}' > 'maxy={self.maxy}'"
            )
        if self.mint > self.maxt:
            raise ValueError(
                f"Bounding box is invalid: 'mint={self.mint}' > 'maxt={self.maxt}'"
            )

    @overload
    def __getitem__(self, key: int) -> float:
        pass

    @overload
    def __getitem__(self, key: slice) -> list[float]:
        pass

    def __getitem__(self, key: int | slice) -> float | list[float]:
        """Index the (minx, maxx, miny, maxy, mint, maxt) tuple.

        Args:
            key: integer or slice object

        Returns:
            the value(s) at that index

        Raises:
            IndexError: if key is out of bounds
        """
        return [self.minx, self.maxx, self.miny, self.maxy, self.mint, self.maxt][key]

    def __iter__(self) -> Iterator[float]:
        """Container iterator.

        Returns:
            iterator object that iterates over all objects in the container
        """
        yield from [self.minx, self.maxx, self.miny, self.maxy, self.mint, self.maxt]

    def __contains__(self, other: BoundingBox) -> bool:
        """Whether or not other is within the bounds of this bounding box.

        Args:
            other: another bounding box

        Returns:
            True if other is within this bounding box, else False

        .. versionadded:: 0.2
        """
        return (
            (self.minx <= other.minx <= self.maxx)
            and (self.minx <= other.maxx <= self.maxx)
            and (self.miny <= other.miny <= self.maxy)
            and (self.miny <= other.maxy <= self.maxy)
            and (self.mint <= other.mint <= self.maxt)
            and (self.mint <= other.maxt <= self.maxt)
        )

    def __or__(self, other: BoundingBox) -> BoundingBox:
        """The union operator.

        Args:
            other: another bounding box

        Returns:
            the minimum bounding box that contains both self and other

        .. versionadded:: 0.2
        """
        return BoundingBox(
            min(self.minx, other.minx),
            max(self.maxx, other.maxx),
            min(self.miny, other.miny),
            max(self.maxy, other.maxy),
            min(self.mint, other.mint),
            max(self.maxt, other.maxt),
        )

    def __and__(self, other: BoundingBox) -> BoundingBox:
        """The intersection operator.

        Args:
            other: another bounding box

        Returns:
            the intersection of self and other

        Raises:
            ValueError: if self and other do not intersect

        .. versionadded:: 0.2
        """
        try:
            return BoundingBox(
                max(self.minx, other.minx),
                min(self.maxx, other.maxx),
                max(self.miny, other.miny),
                min(self.maxy, other.maxy),
                max(self.mint, other.mint),
                min(self.maxt, other.maxt),
            )
        except ValueError:
            raise ValueError(f'Bounding boxes {self} and {other} do not overlap')

    @property
    def area(self) -> float:
        """Area of bounding box.

        Area is defined as spatial area.

        Returns:
            area

        .. versionadded:: 0.3
        """
        return (self.maxx - self.minx) * (self.maxy - self.miny)

    @property
    def volume(self) -> float:
        """Volume of bounding box.

        Volume is defined as spatial area times temporal range.

        Returns:
            volume

        .. versionadded:: 0.3
        """
        return self.area * (self.maxt - self.mint)

    def intersects(self, other: BoundingBox) -> bool:
        """Whether or not two bounding boxes intersect.

        Args:
            other: another bounding box

        Returns:
            True if bounding boxes intersect, else False
        """
        return (
            self.minx <= other.maxx
            and self.maxx >= other.minx
            and self.miny <= other.maxy
            and self.maxy >= other.miny
            and self.mint <= other.maxt
            and self.maxt >= other.mint
        )

    def split(
        self, proportion: float, horizontal: bool = True
    ) -> tuple[BoundingBox, BoundingBox]:
        """Split BoundingBox in two.

        Args:
            proportion: split proportion in range (0,1)
            horizontal: whether the split is horizontal or vertical

        Returns:
            A tuple with the resulting BoundingBoxes

        .. versionadded:: 0.5
        """
        if not (0.0 < proportion < 1.0):
            raise ValueError('Input proportion must be between 0 and 1.')

        if horizontal:
            w = self.maxx - self.minx
            splitx = self.minx + w * proportion
            bbox1 = BoundingBox(
                self.minx, splitx, self.miny, self.maxy, self.mint, self.maxt
            )
            bbox2 = BoundingBox(
                splitx, self.maxx, self.miny, self.maxy, self.mint, self.maxt
            )
        else:
            h = self.maxy - self.miny
            splity = self.miny + h * proportion
            bbox1 = BoundingBox(
                self.minx, self.maxx, self.miny, splity, self.mint, self.maxt
            )
            bbox2 = BoundingBox(
                self.minx, self.maxx, splity, self.maxy, self.mint, self.maxt
            )

        return bbox1, bbox2