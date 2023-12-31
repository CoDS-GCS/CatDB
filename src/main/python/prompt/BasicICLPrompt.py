class BasicICLPrompt(object):
    NUM_EXAMPLE = None
    SEP_EXAMPLE = "\n\n"

    def __init__(self, *args, **kwargs):
        self.example_qualities = []

    def format(self, example: dict):
        # target question
        prompt_target = self.format_target(example)
        return prompt_target
