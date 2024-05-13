from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class BEVDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('background','continuous_lane', 'dashed_lane', 'road_edge'),
        palette=[[0,0,0], [255,0,0], [0,255,0], [0,0,255]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)