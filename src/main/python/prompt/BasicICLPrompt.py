class BasicICLPrompt(object):
    NUM_EXAMPLE = None
    SEP_EXAMPLE = "\n\n"

    def __init__(self, *args, **kwargs):
        self.example_qualities = []

    def format(self, example: dict):
        return self.format_target(example)

    def extra_info(self, examples: dict):
        return self.get_extra_info(examples)