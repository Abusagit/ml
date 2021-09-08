# Here are some tips & lifehacks for more convenient data processing

**Convert categorical variable into dummy/indicator variables - OneHot Encoding:**

```python
X = pd.get_dummies(train_data[features])
```



Finding the best maximum leaf nodes amount in a tree model

```python
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)
```

**Generate random dots**

```python
X, y = sklearn.datasets.make_moons(n_samples=10000, random_state=42, noise=0.1)
```

**Stratified samplinf based on some feature** (*income_category*)

```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, 
                               test_size=0.2, 
                               random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
       strat_train_set = housing.loc[train_index]
       strat_test_set = housing.loc[test_index]

```



**XGBoost**

```path
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
```

<u>early_stopping_rounds</u> -offers a way to automatically find the ideal value for `n_estimators`. Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for `n_estimators`. It's smart to set a high value for `n_estimators` and then use `early_stopping_rounds` to find the optimal time to stop iterating.

<u>Eval_set</u> - When using `early_stopping_rounds`, you also need to set aside some data for calculating the validation scores - this is done by setting the `eval_set` parameter

<u>Learning_rate</u> = Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number (known as the **learning rate**) before adding them in.

This means each tree we add to the ensemble helps us less. So, we can set a higher value for `n_estimators` without overfitting. If we use early stopping, the appropriate number of trees will be determined automatically.

In general, a small learning rate and large number of estimators will yield more accurate XGBoost models, though it will also take the model longer to train since it does more iterations through the cycle. As default, XGBoost sets `learning_rate=0.1`.

<u>n_jobs</u> -  It's common to set the parameter `n_jobs` equal to the number of cores on your machine. On smaller datasets, this won't help



## Mutual information

The trend lines being significantly different from one category to the next indicates an interaction effect.



```
# If the feature has 0.0 values, use np.log1p (log(1+x)) instead of np.log
accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p)
```

![image-20210809175625007](/Users/darji/Library/Application Support/typora-user-images/image-20210809175625007.png)

```
components = [ "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
               "Superplasticizer", "CoarseAggregate", "FineAggregate"]
concrete["Components"] = concrete[components].gt(0).sum(axis=1)

concrete[components + ["Components"]].head(10)
```

Out[6]:

|      | Cement | BlastFurnaceSlag | FlyAsh | Water | Superplasticizer | CoarseAggregate | FineAggregate | Components |
| :--- | :----- | :--------------- | :----- | :---- | :--------------- | :-------------- | :------------ | :--------- |
| 0    | 540.0  | 0.0              | 0.0    | 162.0 | 2.5              | 1040.0          | 676.0         | 5          |
| 1    | 540.0  | 0.0              | 0.0    | 162.0 | 2.5              | 1055.0          | 676.0         | 5          |
| 2    | 332.5  | 142.5            | 0.0    | 228.0 | 0.0              | 932.0           | 594.0         | 5          |
| 3    | 332.5  | 142.5            | 0.0    | 228.0 | 0.0              | 932.0           | 594.0         | 5          |
| 4    | 198.6  | 132.4            | 0.0    | 192.0 | 0.0              | 978.4           | 825.5         | 5          |
| 5    | 266.0  | 114.0            | 0.0    | 228.0 | 0.0              | 932.0           | 670.0         | 5          |
| 6    | 380.0  | 95.0             | 0.0    | 228.0 | 0.0              | 932.0           | 594.0         | 5          |
| 7    | 380.0  | 95.0             | 0.0    | 228.0 | 0.0              | 932.0           | 594.0         | 5          |
| 8    | 266.0  | 114.0            | 0.0    | 228.0 | 0.0              | 932.0           | 670.0         | 5          |
| 9    | 475.0  | 0.0              | 0.0    | 228.0 | 0.0              | 932.0           |               |            |



The `str` accessor lets you apply string methods like `split` directly to columns. The *Customer Lifetime Value* dataset contains features describing customers of an insurance company. From the `Policy` feature, we could separate the `Type` from the `Level` of coverage:

In [7]:

```
customer[["Type", "Level"]] = (  # Create two new features
    customer["Policy"]           # from the Policy feature
    .str                         # through the string accessor
    .split(" ", expand=True)     # by splitting on " "
                                 # and expanding the result into separate columns
)

customer[["Policy", "Type", "Level"]].head(10)
```

Out[7]:

|      | Policy       | Type      | Level |
| :--- | :----------- | :-------- | :---- |
| 0    | Corporate L3 | Corporate | L3    |
| 1    | Personal L3  | Personal  | L3    |
| 2    | Personal L3  | Personal  | L3    |
| 3    | Corporate L2 | Corporate | L2    |
| 4    | Personal L1  | Personal  | L1    |
| 5    | Personal L3  | Personal  | L3    |
| 6    | Corporate L3 | Corporate | L3    |
| 7    | Corporate L3 | Corporate | L3    |
| 8    | Corporate L3 | Corporate | L3    |
| 9    | Special L2   | Special   | L2    |





**GroupBy.transform**

```
customer["AverageIncome"] = (
    customer.groupby("State")  # for each state
    ["Income"]                 # select the income
    .transform("mean")         # and compute its mean
)

customer[["State", "Income", "AverageIncome"]].head(10)
```

| State | Income     | AverageIncome |              |
| :---- | :--------- | :------------ | ------------ |
| 0     | Washington | 56274         | 38122.733083 |
| 1     | Arizona    | 0             | 37405.402231 |
| 2     | Nevada     | 48767         | 38369.605442 |
| 3     | California | 0             | 37558.946667 |
| 4     | Washington | 43836         | 38122.733083 |
| 5     | Oregon     | 62902         | 37557.283353 |
| 6     | Oregon     | 55350         | 37557.283353 |
| 7     | Arizona    | 0             | 37405.402231 |
| 8     | Oregon     | 14072         | 37557.283353 |
| 9     | Oregon     | 28812         | 37557.283353 |

