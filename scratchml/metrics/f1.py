from scratchml.metrics.f_beta import FBeta

class F1(FBeta):
    def __init__(self):
        super().__init__(beta=1.0)