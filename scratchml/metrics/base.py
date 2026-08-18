class BaseMetric:
    def error(self, y_true, y_pred):
        return y_pred - y_true

    def score(self, y_true, y_pred):
        raise NotImplementedError