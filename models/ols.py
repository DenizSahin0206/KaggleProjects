import statsmodels.api as sm


class OLS:
    def __init__(self):
        self.model = sm.OLS()

    def fit(self, x, y):
        x = sm.add_constant(x)
        self.model.fit(y, x)

    def predict(self, x):
        return self.model.predict(x)

    def model_summary(self):
        return self.model.summary()
