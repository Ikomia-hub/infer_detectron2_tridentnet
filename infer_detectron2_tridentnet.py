from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from infer_detectron2_tridentnet.infer_detectron2_tridentnet_process import TridentnetFactory
        # Instantiate process object
        return TridentnetFactory()

    def getWidgetFactory(self):
        from infer_detectron2_tridentnet.infer_detectron2_tridentnet_widget import TridentnetWidgetFactory
        # Instantiate associated widget object
        return TridentnetWidgetFactory()
