from infer_detectron2_tridentnet import update_path
from ikomia import core, dataprocess
import copy
import os
import numpy
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from infer_detectron2_tridentnet.TridentNet_git.tridentnet import add_tridentnet_config


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class TridentnetParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.cuda = True
        self.proba = 0.8
        self.update = False

    def setParamMap(self, param_map):
        self.cuda = int(param_map["cuda"])
        self.proba = int(param_map["proba"])

    def getParamMap(self):
        param_map = core.ParamMap()
        param_map["cuda"] = str(self.cuda)
        param_map["proba"] = str(self.proba)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class Tridentnet(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        if param is None:
            self.setParam(TridentnetParam())
        else:
            self.setParam(copy.deepcopy(param))

        # get and set config model
        self.folder = os.path.dirname(os.path.realpath(__file__))
        self.MODEL_NAME_CONFIG = "tridentnet_fast_R_101_C4_3x"
        self.cfg = None
        self.predictor = None
        self.class_names = None
        self.colors = None

        # add output
        self.addOutput(dataprocess.CObjectDetectionIO())

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 2

    def run(self):
        self.beginTaskRun()

        # we use seed to keep the same color for our masks + boxes + labels (same random each time)
        numpy.random.seed(30)

        # Get input :
        img_input = self.getInput(0)
        src_image = img_input.getImage()

        # Get output :
        output_image = self.getOutput(0)
        obj_detect_out = self.getOutput(1)
        obj_detect_out.init("TridentNet", 0)

        # Get parameters :
        param = self.getParam()

        # instantiate or update predictor
        if param.update or self.predictor is None:
            self.cfg = get_cfg()
            add_tridentnet_config(self.cfg)
            self.cfg.merge_from_file(self.folder + "/TridentNet_git/configs/" + self.MODEL_NAME_CONFIG + ".yaml")
            self.cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/TridentNet/" \
                                     "tridentnet_fast_R_101_C4_3x/148572198/model_final_164568.pkl"
            self.cfg.MODEL.DEVICE = "cuda" if param.cuda else "cpu"
            self.predictor = DefaultPredictor(self.cfg)
            self.class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes")
            self.colors = numpy.array(numpy.random.randint(0, 255, (len(self.class_names), 3)))
            self.colors = [[int(c[0]), int(c[1]), int(c[2])] for c in self.colors]
            param.update = False

        outputs = self.predictor(src_image)

        # get outputs instances
        output_image.setImage(src_image)
        boxes = outputs["instances"].pred_boxes
        scores = outputs["instances"].scores
        classes = outputs["instances"].pred_classes

        # to numpy
        if param.cuda:
            boxes_np = boxes.tensor.cpu().numpy()
            scores_np = scores.cpu().numpy()
        else:
            boxes_np = boxes.tensor.numpy()
            scores_np = scores.numpy()

        self.emitStepProgress()

        # Show Boxes + labels
        for i in range(len(scores_np)):
            if scores_np[i] > param.proba:
                box_x = float(boxes_np[i][0])
                box_y = float(boxes_np[i][1])
                box_w = float(boxes_np[i][2] - boxes_np[i][0])
                box_h = float(boxes_np[i][3] - boxes_np[i][1])
                obj_detect_out.addObject(i, self.class_names[classes[i]], float(scores_np[i]),
                                         box_x, box_y, box_w, box_h, self.colors[classes[i]])

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class TridentnetFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_detectron2_tridentnet"
        self.info.shortDescription = "TridentNet inference model of Detectron2 for object detection."
        self.info.description = "TridentNet inference model for object detection trained on COCO. " \
                                "Implementation from Detectron2 (Facebook Research). " \
                                "Trident Network (TridentNet) aims to generate scale-specific feature maps " \
                                "with a uniform representational power. We construct a parallel multi-branch " \
                                "architecture in which each branch shares the same transformation parameters " \
                                "but with different receptive fields. TridentNet-Fast is a fast approximation " \
                                "version of TridentNet that could achieve significant improvements without " \
                                "any additional parameters and computational cost." \
                                "This Ikomia plugin can make inference of pre-trained model " \
                                "with ResNet101 backbone + C4 head."
        self.info.authors = "Li, Yanghao and Chen, Yuntao and Wang, Naiyan and Zhang, Zhaoxiang"
        self.info.article = "Scale-Aware Trident Networks for Object Detection"
        self.info.journal = "IEEE International Conference on Computer Vision (ICCV)"
        self.info.year = 2019
        self.info.license = "Apache-2.0 License"
        self.info.documentationLink = "https://detectron2.readthedocs.io/index.html"
        self.info.repo = "https://github.com/facebookresearch/detectron2/tree/master/projects/TridentNet"
        self.info.path = "Plugins/Python/Detection"
        self.info.iconPath = "icons/detectron2.png"
        self.info.version = "1.3.1"
        self.info.keywords = "object,facebook,detectron2,detection,multi,scale"

    def create(self, param=None):
        # Create process object
        return Tridentnet(self.info.name, param)
