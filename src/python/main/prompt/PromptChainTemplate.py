from BasicPrompt import BasicPrompt


class DataPreprocessingChainPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        BasicPrompt.__init__(self, *args, **kwargs)
        self.rules = []
        self.question = "TODO"

