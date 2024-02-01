class BasicICLPrompt(object):
    NUM_EXAMPLE = None
    SEP_EXAMPLE = "\n\n"

    def __init__(self, *args, **kwargs):
        self.example_qualities = []

    def format(self, examples: dict):
        return self.format_target(examples)