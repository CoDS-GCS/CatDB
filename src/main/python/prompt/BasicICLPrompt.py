class BasicICLPrompt(object):
    NUM_EXAMPLE = None
    SEP_EXAMPLE = "\n\n"

    def __init__(self, *args, **kwargs):
        self.example_qualities = []

    def format(self):
        return self.format_target()

    def extra_info(self):
        return self.get_extra_info()