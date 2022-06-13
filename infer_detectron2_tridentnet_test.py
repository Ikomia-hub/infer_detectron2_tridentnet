import logging
import cv2
from ikomia.utils.tests import run_for_test

logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info("===== Test::infer detectron2 tridentnet =====")
    logger.info("----- Use default parameters")
    img = cv2.imread(data_dict["images"]["detection"]["coco"])
    input_img_0 = t.getInput(0)
    input_img_0.setImage(img)
    return run_for_test(t)