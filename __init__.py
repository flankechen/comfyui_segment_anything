from .node import *
from .install import *

NODE_CLASS_MAPPINGS = {
    'SAMModelLoader (segment anything)': SAMModelLoader,
    'GroundingDinoModelLoader (segment anything)': GroundingDinoModelLoader,
    'GroundingDinoSAMSegment (segment anything)': GroundingDinoSAMSegment,
    'InvertMask (segment anything)': InvertMask,
    "IsMaskEmpty": IsMaskEmptyNode,
    "LoadImagePathSAM": LoadImagePath,
    "SaveImageSAM": SaveImagePath,
    'GroundingDinoBbox': GroundingDinoBbox,
    'GroundingDinoBboxSingle': GroundingDinoBboxSingle,
    'GroundingDinoBboxCatHead': GroundingDinoBboxCatHead,
}

__all__ = ['NODE_CLASS_MAPPINGS']


