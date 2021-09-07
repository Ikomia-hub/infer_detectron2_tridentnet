from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class Detectron2_TridentNet(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from Detectron2_TridentNet.Detectron2_TridentNet_process import Detectron2_TridentNetProcessFactory
        # Instantiate process object
        return Detectron2_TridentNetProcessFactory()

    def getWidgetFactory(self):
        from Detectron2_TridentNet.Detectron2_TridentNet_widget import Detectron2_TridentNetWidgetFactory
        # Instantiate associated widget object
        return Detectron2_TridentNetWidgetFactory()
