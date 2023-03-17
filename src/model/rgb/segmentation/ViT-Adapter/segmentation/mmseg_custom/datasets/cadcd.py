import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

CSLabels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
]

@DATASETS.register_module()
class CADCDataset(CustomDataset):
    """CADC dataset.
    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for CADC dataset.
    """

    CLASSES = ('road')

    PALETTE = [[128, 64, 128]]

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelTrainIds.png',
                 **kwargs):
        super(CADCDataset, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

    @staticmethod
    def _convert_to_label_id(result):
        """Convert trainId to id for CADC."""
        if isinstance(result, str):
            result = np.load(result)
        result_copy = result.copy()
        for trainId, label in CSLabels.trainId2label.items():
            result_copy[result == trainId] = label.id

        return result_copy

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.
        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.
        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):
            if to_label_id:
                result = self._convert_to_label_id(result)
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
            for label_id, label in CSLabels.id2label.items():
                palette[label_id] = label.color

            output.putpalette(palette)
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for Cityscapes
        evaluation).
        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.
        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)

        return result_files

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 imgfile_prefix=None):
        """Evaluation in CADC/default protocol.
        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file,
                for cityscapes evaluation only. It includes the file path and
                the prefix of filename, e.g., "a/b/prefix".
                If results are evaluated with cityscapes protocol, it would be
                the prefix of output png files. The output files would be
                png images under folder "a/b/prefix/xxx.png", where "xxx" is
                the image name of cityscapes. If not specified, a temp file
                will be created for evaluation.
                Default: None.
        Returns:
            dict[str, float]: CADC/default metrics.
        """

        eval_results = dict()
        metrics = metric.copy() if isinstance(metric, list) else [metric]
        if len(metrics) > 0:
            eval_results.update(
                super(CADCDataset,
                      self).evaluate(results, metrics, logger))

        return eval_results