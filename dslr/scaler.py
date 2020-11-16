class StandardScaler:
	def __init__(self):
		self.mean = None
		self.std = None
		pass

	def std_val(self, X, mean):
		sum_val = 0
		for x in X:
			sum_val += (x - mean) ** 2
		return (sum_val / (len(X) - 1)) ** 0.5

	def mean_val(self, X):
		sum_val = 0
		for x in X:
			sum_val += x
		return sum_val / len(X)

	def fit(self, X):
		self.mean, self.std = [], []
		for i, col in enumerate(X.columns):
			self.mean.append(self.mean_val(X[col]))
			self.std.append(self.std_val(X[col], self.mean[i]))
		pass

	def transform(self, X):
		self.fit(X)
		X_temp = X.copy()
		for i, col in enumerate(X_temp.columns):
			vals_scaled = []
			for x in X_temp[col]:
				vals_scaled.append((x - self.mean[i]) / self.std[i])
			X_temp[col] = vals_scaled
		return X_temp
		pass

	def fit_transform(self, X):
		self.fit(X)
		X_temp = X.copy()
		for i, col in enumerate(X_temp.columns):
			vals_scaled = []
			for x in X_temp[col]:
				vals_scaled.append((x - self.mean[i]) / self.std[i])
			X_temp[col] = vals_scaled
		return X_temp
		pass
