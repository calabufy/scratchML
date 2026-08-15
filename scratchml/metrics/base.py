class BaseMetric:
    def score(self, y_true, y_pred):
        raise NotImplementedError