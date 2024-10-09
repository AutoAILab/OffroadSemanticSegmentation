from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class Rellis3DDataset(CustomDataset):

    """RELLIS dataset.
- 0: void
  1: dirt
  3: grass
  4: tree
  5: pole
  6: water
  7: sky
  8: vehicle
  9: object
  10: asphalt
  12: building
  15: log
  17: person
  18: fence
  19: bush
  23: concrete
  27: barrier
  31: puddle
  33: mud
  34: rubble
    """

# {
#     1,
#     3,
#     4,
#     5,
#     6,
#     7,
#     8,
#     9,
#     10,
#     12,
#     15,
#     17,
#     18,
#     19,
#     23,
#     27,
#     31,
#     33,
#     34
# }

    # CLASSES = ("void", "dirt", "grass", "tree", "pole", "water", "sky", "vehicle",
    CLASSES = ("dirt", "grass", "tree", "pole", "water", "sky", "vehicle", 
            "object", "asphalt", "building", "log", "person", "fence", "bush", 
            "concrete", "barrier", "puddle", "mud", "rubble")

    # PALETTE = [[0, 0, 0], [108, 64, 20], [0, 102, 0], [0, 255, 0], [0, 153, 153],
    PALETTE = [[108, 64, 20], [0, 102, 0], [0, 255, 0], [0, 153, 153], 
            [0, 128, 255], [0, 0, 255], [255, 255, 0], [255, 0, 127], [64, 64, 64], 
            [255, 0, 0], [102, 0, 0], [204, 153, 255], [102, 0, 204], [255, 153, 204], 
            [170, 170, 170], [41, 121, 255], [134, 255, 239], [99, 66, 34], [110, 22, 138]]

    def __init__(self, **kwargs):
        super(Rellis3DDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            # reduce_zero_label = False,
            **kwargs)
        self.CLASSES = ("dirt", "grass", "tree", "pole", "water", "sky", "vehicle", 
            "object", "asphalt", "building", "log", "person", "fence", "bush", 
            "concrete", "barrier", "puddle", "mud", "rubble")
        self.PALETTE = [[108, 64, 20], [0, 102, 0], [0, 255, 0], [0, 153, 153], 
            [0, 128, 255], [0, 0, 255], [255, 255, 0], [255, 0, 127], [64, 64, 64], 
            [255, 0, 0], [102, 0, 0], [204, 153, 255], [102, 0, 204], [255, 153, 204], 
            [170, 170, 170], [41, 121, 255], [134, 255, 239], [99, 66, 34], [110, 22, 138]]
        
    # def __init__(self, **kwargs):
    #     super(Rellis3DDataset, self).__init__(
    #         img_suffix='.jpg',
    #         seg_map_suffix='.png',
    #         reduce_zero_label=False,
    #         **kwargs)