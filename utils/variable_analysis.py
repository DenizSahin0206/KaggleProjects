import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
pd.set_option("Display.max_columns", None)
pd.set_option("Display.max_rows", None)
plt.style.use('ggplot')

"""
--- DATA UNDERSTANDING ----
Variable analysis to check the relationship with the variables and the house prices:
"""

df_train = pd.read_csv(
    r"/Users/teleradio/Desktop/GitHub/MachineLearning/HousePricingUSA/data/train.csv"
)

df_test = pd.read_csv(
    r"/Users/teleradio/Desktop/GitHub/MachineLearning/HousePricingUSA/data/test.csv"
)

#       ---- STEP 1: DATA UNDERSTANDING ----
print(f"Number of rows and columns:\n{df_train.shape}\n\n")       # Number of rows and columns
print(f"First rows:\n{df_train.head(5)}\n\n")                     # Check first 5 rows
print(f"Columns:\n{df_train.columns}\n\n")                        # Check column names (in a list)
print(f"Datatypes:\n{df_train.dtypes}\n\n")                       # Check the datatypes of the columns
print(f"Summarizing statistics:\n{df_train.describe()}\n\n")      # Show summarizing statistics on all variables (columns)
print(f"NullValues:\n{df_train.isnull().sum()}\n\n")              # Check the number of null values

#       ---- STEP 2: DATA PREPARATION ----
# Analyze importance of variables
# Select only the numeric columns to calculate the correlations
numeric_columns = df_train.select_dtypes(include=[float, int]).columns
numeric_df_train = df_train[numeric_columns]
correlation_matrix = numeric_df_train.corr()
high_correlation = correlation_matrix[(correlation_matrix < -0.3) | (correlation_matrix > 0.5)]
sales_price_correlation = correlation_matrix['SalePrice'].abs().sort_values(ascending=False)
print(len(numeric_df_train.columns))
print(f"{sales_price_correlation}")
# Create a correlation heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(
    high_correlation,
    annot=True,
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Heatmap")
plt.show()

# Only keep relevant columns/variables
df_train = df_train[[
    'Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
    'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
    'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
    'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
    'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
    'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
    'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
    'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
    'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndf_trainlrSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
    'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
    'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
    'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
    'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
    'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
    'SaleCondition', 'SalePrice'
]]

# ---- First check the distribution of the house prices ----
print(
    f"SalesPrice:\n{df_train.describe()}\n\n"       # Check summarizing statistics

)

# Select only the numeric columns
numeric_columns = df_train.select_dtypes(include=[float, int]).columns
numeric_df_train = df_train[numeric_columns]
correlation_matrix = numeric_df_train.corr()

# Create a correlation heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()
