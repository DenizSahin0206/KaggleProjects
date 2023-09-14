from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV


class DataPreparation:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def drop_id(self):
        return 0

    def add_constant(self):
        scaler = StandardScaler()
        self.x = scaler.fit_transform(self.x) + 1
        return self.x
