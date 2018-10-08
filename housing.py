import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT= "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH= "datasets/housing"
HOUSING_URL= DOWNLOAD_ROOT+HOUSING_PATH+"/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path= os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz= tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
    
#fetch_housing_data()

import pandas as pd

#function to load the data
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path= os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

housing= load_housing_data()
print(housing.head())
print(housing.info())
print(housing['ocean_proximity'].value_counts())
print(housing.describe())

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# creating a test set
import numpy as np
def split_train_test(data, test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size= int(len(data)* test_ratio)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

#to make test data same when we insert an update to it or run the above function
    
import hashlib

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1]<256 *test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids=data[id_column]
    in_test_set= ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]
'''
housing_with_id= housing.reset_index() #adds an 'index' column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
housing_with_id["id"]= housing["longitude"] + 1000+ housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id,0.2,"id")
'''    
from sklearn.model_selection import train_test_split
train_set, test_set= train_test_split(housing, test_size=0.2, random_state= 42)

housing["income_cat"]= np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

#Now we are redy to do stratified sampling based on the income category.

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set =housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
print(housing["income_cat"].value_counts()/len(housing))

for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

#making a copy of the training set to work upon    
housing= strat_train_set.copy()

#Visualizing Geographical Data
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population",
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()

corr_matrix= housing.corr()


from pandas.tools.plotting import scatter_matrix

attributes= ["median_house_value", "median_income", "total_rooms",
             "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix= housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

#preparing the Data for Machine Learning Algorithms

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

from sklearn.preprocessing import Imputer

imputer= Imputer(strategy="median")
housing_num= housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)
print(imputer.statistics_)
print(housing_num.median().values)

#Now, using this "trained" imputer to transform the training set by replacing missing values by the learned median:
X= imputer.transform(housing_num)

#The result is a plain Numpy array containing the transformed features. If you want to put it back into a
#Pandas DataFrame, it’s simple:
housing_tr =pd.DataFrame(X, columns=housing_num.columns)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat= housing["ocean_proximity"]
housing_cat_encoded= encoder.fit_transform(housing_cat)
print(encoder.classes_)

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot= encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
print(housing_cat_1hot)
housing_cat_1hot.toarray()

'''We can apply both transformations (from text categories to integer categories, then from integer categories
to one-hot vectors) in one shot using the LabelBinarizer class:'''

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)


from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


#Transformation Piplines for our all the data transformation steps
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline= Pipeline([('imputer', Imputer(strategy="median")),
                        ('attribs_adder', CombinedAttributesAdder()),
                        ('std_scaler', StandardScaler()),
                    ])
housing_num_tr= num_pipeline.fit_transform(housing_num)


from sklearn.base import BaseEstimator, TransformerMixin
class LabelBinarizerPipelineFriendly(LabelBinarizer):
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(LabelBinarizerPipelineFriendly, self).fit(X)
    def transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).transform(X)

    def fit_transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

from sklearn.pipeline import FeatureUnion

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizerPipelineFriendly()),
    ])

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline)
    ])
                
housing_prepared = full_pipeline.fit_transform(housing)


#Training and Evaluating on the Training Set
#Linear Regression Model
from sklearn.linear_model import LinearRegression

lin_reg= LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#you now have a working Linear Regression model. Let's try it out on a few instances from the training set:

some_data= housing.iloc[:5]
some_labels= housing_labels.iloc[:5]
some_data_prepared= full_pipeline.transform(some_data)
print("Predictions:\t", lin_reg.predict(some_data_prepared))
print("Labels:\t\t", list(some_labels))


from sklearn.metrics import mean_squared_error
housing_predictions= lin_reg.predict(housing_prepared)
lin_mse= mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

#Decision Tree Regressor Model
from sklearn.tree import DecisionTreeRegressor

tree_reg= DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse= mean_squared_error(housing_labels, housing_predictions)
tree_rmse =np.sqrt(tree_mse)
print("tree_rmse= ",tree_rmse)
#rmse value is 0.0, it means our model is absolutely perfect? No, the model is overfitted
'''
A great alternative is to use Scikit-Learn’s cross-validation feature. The following code performs K-fold
cross-validation: it randomly splits the training set into 10 distinct subsets called folds, then it trains and
evaluates the Decision Tree model 10 times, picking a different fold for evaluation every time and
training on the other 9 folds. The result is an array containing the 10 evaluation scores:
'''

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

def display_scores(scores_):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation", scores.std())
    

display_scores(tree_rmse)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

from sklearn.ensemble import RandomForestRegressor
'''
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
display_scores(forest_rmse_scores)
'''

from sklearn.model_selection import GridSearchCV
param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
        ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')

grid_search.fit(housing_prepared, housing_labels)
print(grid_search.best_params_)
print(grid_search.best_estimator_)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


#Evaluate Your System on the Test Set
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) # => evaluates to 48,209.6