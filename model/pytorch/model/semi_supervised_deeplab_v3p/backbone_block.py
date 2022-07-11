"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports

# Internal imports
from model.pytorch.base_block import BaseBlock
from model.pytorch.core.backbone.resnet_backbone import resnet50, IntermediateLayerGetter


class BackboneBlock(BaseBlock):

    """
        Example Network Block used to define separate parts of the network
    """

    def __init__(self, name="backbone"):
        """
        Initialise the variables in the block
        """
        super().__init__(name=name)

        # Parameters
        self.return_layers = {'layer4': 'out', 'layer1': 'low_level'}

        # Block variables
        self.backbone = None

    def build(self):
        """
        Build the actual network graph
        """
        self.backbone = resnet50(pretrained=True, replace_stride_with_dilation=[False, False, True], max_pool_stem=False)
        self.backbone = IntermediateLayerGetter(self.backbone, return_layers=self.return_layers)

    def forward(self, x):
        """
        Define the forward function that will feed data through the network graph
        """
        x = self.backbone(x)
        return x
