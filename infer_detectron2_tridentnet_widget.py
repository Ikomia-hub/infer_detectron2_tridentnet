from ikomia import utils, core, dataprocess
from ikomia.utils import qtconversion
from infer_detectron2_tridentnet.infer_detectron2_tridentnet_process import TridentnetParam
# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits core.CProtocolTaskWidget from Ikomia API
# --------------------
class TridentnetWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = TridentnetParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()

        # cuda parameter
        cuda_label = QLabel("CUDA")
        self.cuda_ckeck = QCheckBox()
        self.cuda_ckeck.setChecked(True)

        # proba parameter
        proba_label = QLabel("Threshold :")
       
        self.proba_spinbox = QDoubleSpinBox()
        self.proba_spinbox.setValue(0.8)
        self.proba_spinbox.setSingleStep(0.1)
        self.proba_spinbox.setMaximum(1)
        if self.parameters.proba != 0.8:
            self.proba_spinbox.setValue(self.parameters.proba)

        self.gridLayout.setColumnStretch(0,0)
        self.gridLayout.addWidget(self.cuda_ckeck, 0, 0)
        self.gridLayout.setColumnStretch(1,1)
        self.gridLayout.addWidget(cuda_label, 0, 1)
        self.gridLayout.addWidget(proba_label, 1, 0)
        self.gridLayout.addWidget(self.proba_spinbox, 1, 1)
        self.gridLayout.setColumnStretch(2,2)

        # Set widget layout
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)
        self.set_layout(layout_ptr)

        if not self.parameters.cuda:
            self.cuda_ckeck.setChecked(False)

    def on_apply(self):
        # Apply button clicked slot
        if self.cuda_ckeck.isChecked():
            self.parameters.cuda = True
        else:
            self.parameters.cuda = False
        self.parameters.proba = self.proba_spinbox.value()
        self.parameters.update = True
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits dataprocess.CWidgetFactory from Ikomia API
# --------------------
class TridentnetWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_detectron2_tridentnet"

    def create(self, param):
        # Create widget object
        return TridentnetWidget(param, None)
