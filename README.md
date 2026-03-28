# From-Scratch Housing Price Regression and Model Comparison

This project was developed to reinforce my understanding of the mathematical and algorithmic content of Andrew Ng's Stanford Machine Learning Specialization by reimplementing core supervised learning methods from first principles. The notebook centers on housing-price regression using a Kaggle `Housing.csv` dataset, with custom NumPy implementations of normalization, cost computation, gradient computation, and batch gradient descent, followed by carefully documented comparisons against several scikit-learn baselines. The result is both a mathematical study and an engineering exercise in turning theory into working code.

## Motivation

The central goal of this project was not simply to fit a predictive model, but to reproduce the mechanics of regression learning at the level of equations, derivatives, and iterative optimization. Rather than relying on `sklearn` to solve the core regression problem end to end, I implemented the principal learning routines manually so that each update step, cost reduction, and performance tradeoff could be inspected directly.

More specifically, the notebook was designed to:

- reimplement the core regression workflow from scratch in NumPy
- connect the mathematical derivations from the Stanford specialization to executable code
- observe optimization behavior through explicit cost histories and residual plots
- compare hand-built models against stronger library-based baselines without confusing those baselines with the core learning implementation

## Project Scope

The notebook is intentionally split between from-scratch learning code and carefully bounded use of library utilities.

### Implemented from scratch in NumPy

- Z-score feature normalization
- mean-squared-error cost computation
- batch gradient computation for multivariate linear regression
- batch gradient descent for multivariate linear regression
- L2-regularized cost computation
- L2-regularized gradient computation
- L2-regularized gradient descent
- manual prediction via `np.dot(X, w) + b`

### Used from scikit-learn for preprocessing, comparison, or evaluation

- `OneHotEncoder` for `furnishingstatus`
- `train_test_split`
- `PolynomialFeatures`
- `RFE`
- `StandardScaler`
- `r2_score`, `mean_squared_error`, `mean_absolute_error`
- `KNeighborsRegressor`
- `GradientBoostingRegressor`
- `GridSearchCV`

## Dataset

The notebook uses a Kaggle housing-price regression dataset stored as `Housing.csv`. The learning problem is to predict house price from a mixture of numerical and categorical attributes describing property size, room counts, amenities, access, location preference, parking capacity, and furnishing status.

### Raw schema shown in the notebook

- `price`
- `area`
- `bedrooms`
- `bathrooms`
- `stories`
- `mainroad`
- `guestroom`
- `basement`
- `hotwaterheating`
- `airconditioning`
- `parking`
- `prefarea`
- `furnishingstatus`

### Dataset preparation summary

| Stage | Shape / Result |
| --- | --- |
| Initial dataset | `(545, 15)` |
| After one-hot encoding and dropping original `furnishingstatus` | still 15 columns including target |
| After IQR-based outlier removal | `(530, 15)` |
| Train/test split | `371` training rows, `159` testing rows |
| Final learning target | `price / 100000` as `float` |
| Final feature count | `14` input features |

The notebook also verifies that, after preprocessing, the dataset contains:

- no remaining string-valued columns
- no null values
- no duplicate rows

## Preprocessing and Exploratory Analysis

The preprocessing sequence is explicit and fully shown in the notebook:

1. Load `Housing.csv` from Google Drive inside Colab.
2. One-hot encode `furnishingstatus` into:
   - `furnishingstatus_furnished`
   - `furnishingstatus_semi-furnished`
   - `furnishingstatus_unfurnished`
3. Replace string-valued binary fields with numeric `1/0`.
4. Drop the original `furnishingstatus` column because its information has been represented in encoded form.
5. Compute a full correlation matrix.
6. Generate feature-vs-price scatter plots for every predictor.
7. Remove price outliers using the IQR rule.
8. Convert `price` to `float` and scale it down by dividing by `100000`.
9. Check for residual strings, nulls, and duplicates.
10. Split the data into training and testing sets with a `70/30` partition.
11. Normalize features with Z-score scaling.
12. Convert pandas objects to NumPy arrays for the custom optimization routines.

### Correlation results shown in the notebook

The notebook prints the full correlation matrix. The direct correlations with `price` are:

| Feature | Correlation with `price` |
| --- | ---: |
| `area` | `0.535997` |
| `bathrooms` | `0.517545` |
| `airconditioning` | `0.452954` |
| `stories` | `0.420712` |
| `parking` | `0.384394` |
| `bedrooms` | `0.366494` |
| `prefarea` | `0.329777` |
| `mainroad` | `0.296898` |
| `guestroom` | `0.255517` |
| `furnishingstatus_furnished` | `0.229350` |
| `basement` | `0.187057` |
| `hotwaterheating` | `0.093073` |
| `furnishingstatus_semi-furnished` | `0.063656` |
| `furnishingstatus_unfurnished` | `-0.280587` |

These outputs indicate that `area`, `bathrooms`, `airconditioning`, and `stories` are among the strongest positive correlates of price in the processed dataset, while `furnishingstatus_unfurnished` is negatively correlated with price.

## Algorithms and Mathematical Formulations

### 1. Z-score normalization

The notebook defines the normalization step as:

$$
Z = \frac{X - \mu}{\sigma}
$$

Implementation details:

- `zscore_normalize_features(X)` computes `mu = np.mean(X, axis=0)` and `sigma = np.std(X, axis=0)`.
- It applies vectorized broadcasting to produce `(X - mu) / sigma`.
- The notebook visualizes the transformation for `area`, `parking`, and `mainroad` using paired histograms of original and normalized values.
- The same normalization function is applied separately to `X_train` and `X_test` as written in the notebook.

### 2. Multivariate linear regression with batch gradient descent

The notebook introduces the regression objective with the following equations:

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{wb}(x^{(i)}) - y^{(i)} \right)^2
$$
</br><br>

$$
f_{wb}(x^{(i)}) = w \cdot x + b
$$

The derivatives used for optimization are copied below exactly as they appear in the notebook:

$$
\frac{dJ(w, b)}{dw} = \frac{1}{m} \sum_{i=0}^{m-1} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) x^{(i)}
$$

$$
\frac{dJ(w, b)}{db} = \frac{1}{m} \sum_{i=0}^{m-1} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)
$$

The update rule is:

$$
\text{repeat until convergence:} \left\{
\begin{aligned}
  w & = w - \alpha \frac{\partial J(w, b)}{\partial w} \\
  b & = b - \alpha \frac{\partial J(w, b)}{\partial b}
\end{aligned}
\right.
$$
$\alpha$ is the learning rate

Implementation details:

- `compute_cost(X, y, w, b)` loops over the `m` training examples, computes `np.dot(X[i], w) + b`, accumulates squared error, and divides by `2m`.
- `compute_gradient(X, y, w, b)` uses nested loops over examples and features to accumulate `dj_dw` and `dj_db`.
- `gradient_descent(...)` deep-copies the weight vector, updates `w` and `b` iteratively, stores `J_history`, and prints progress every `num_iters / 10`.
- The model is initialized with `w = 0`, `b = 0`, `alpha = 0.01`, and `iterations = 1000`.

### 3. Polynomial regression

The notebook explains the move to polynomial regression as follows: `PolynomialFeatures` is used to raise the features to higher-order powers, effectively changing the model from linear regression to polynomial regression to capture non-linearity in the data.

Implementation details:

- `PolynomialFeatures(degree=2)` is used to expand `X_train_norm` and `X_test_norm`.
- The optimization over the expanded design matrix is still performed by the custom NumPy implementation of `compute_cost`, `compute_gradient`, and `gradient_descent`.
- This makes the basis expansion library-based, but the learning algorithm itself remains the notebook author's own implementation.

### 4. L2-regularized regression

The notebook gives the regularized objective exactly as:

$$
J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2  + \frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2
$$
</br><br>

$$
f_{wb}(x^{(i)}) = w \cdot x + b
$$

The corresponding derivatives are:

$$
\frac{dJ(w, b)}{dw} = \frac{1}{m} \sum_{i=0}^{m-1} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) x^{(i)} + \frac{\lambda}{m} w_j
$$

$$
\frac{dJ(w, b)}{db} = \frac{1}{m} \sum_{i=0}^{m-1} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)
$$

And the update rule is written as:

$$
\text{repeat until convergence:} \left\{
\begin{aligned}
  w & = w - \alpha \frac{\partial J(w, b)}{\partial w} \\
  b & = b - \alpha \frac{\partial J(w, b)}{\partial b}
\end{aligned}
\right.
$$

Implementation details:

- `compute_cost_reg(X, y, w, b, lambda_=1)` adds the L2 penalty term to the base MSE cost.
- `compute_gradient_reg(X, y, w, b, lambda_)` first computes the ordinary gradients, then adds `(lambda_/m) * w[j]` to each component of `dj_dw`.
- `gradient_descent_reg(...)` mirrors the unregularized version while calling the regularized gradient and cost functions.
- Regularization is applied both to the polynomial model and to the original linear model.

### 5. Recursive Feature Elimination (RFE)

The notebook uses RFE as a feature selection stage before retraining the custom regularized regression pipeline.

Implementation details:

- `LinearRegression()` is used as the estimator for `RFE`.
- `n_features_to_select=9`.
- The support mask printed by the notebook is:
  - `[ True False  True  True  True False  True False  True  True  True False False  True]`
- The ranking vector printed by the notebook is:
  - `[1 6 1 1 1 3 1 2 1 1 1 5 4 1]`
- The selected features correspond to:
  - `area`
  - `bathrooms`
  - `stories`
  - `mainroad`
  - `basement`
  - `airconditioning`
  - `parking`
  - `prefarea`
  - `furnishingstatus_unfurnished`

The notebook then trains:

- a regularized linear model on the selected features
- a degree-2 polynomial model on the selected features, again optimized with the custom regularized gradient-descent implementation

### 6. K-nearest neighbors comparison

The notebook includes a nonparametric baseline for comparison.

Implementation details:

- `StandardScaler` is applied to `X_train` and `X_test`.
- `KNeighborsRegressor` is trained for every `k` from `1` to `99`.
- Training and testing `R^2` are recorded for each `k`.
- The best test result occurs at `k = 13`.

### 7. Gradient Boosting comparison and tuning

The notebook includes an ensemble baseline and then tunes it.

Implementation details:

- A default `GradientBoostingRegressor()` is trained first.
- `staged_predict(X_train)` is used to build a training-MSE history across boosting iterations.
- A second model is tuned with `GridSearchCV` over:
  - `n_estimators`: `[10,20,30,40,50,60,70,80,90,100, 150,200,300]`
  - `learning_rate`: `[0.00001,0.0001,0.001,0.01, 0.05, 0.1]`
  - `max_depth`: `[2,3,4,5]`
- The tuned search uses `cv=5`, `scoring='r2'`, `n_jobs=-1`, `verbose=1`.

## Evaluation Metrics Used

The notebook explicitly states the following metrics:

$$\text{Mean Squared Error} = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

$$\text{Normalized Mean Squared Error} = \frac{\sqrt{\text{MSE}}}{\text{Range of } y}$$

$$ \text{Range of y} = y(max) - y(min) $$

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
$$

Get the $R^2$ metric of our model
$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

The notebook also notes that `R^2` is ultimately computed with the built-in scikit-learn method.

## Results

### Comprehensive model comparison

| Model | Train `R^2` | Test `R^2` | MAE | Average Error (%) | MSE | NRMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear Regression | `0.6541` | `0.6824` | `6.5145` | `8.86` | `69.9115` | `0.113759` |
| Polynomial Regression | `0.7712` | `0.5097` | `8.0172` | `10.91` | `107.9243` | `0.141342` |
| Polynomial Regression Regulated | `0.7218` | `0.6272` | `7.2184` | `9.82` | `82.0684` | `0.141342` printed in notebook |
| Linear Regression Regulated | `0.6432` | `0.6939` | `6.2286` | `8.47` | `67.3720` | `0.111674` |
| Linear Regression RFE selection | `0.6306` | `0.6687` | `6.3632` | `8.66` | `72.9296` | `0.116189` |
| Polynomial Regression RFE selection | `0.6507` | `0.6425` | `6.8816` | `9.36` | `78.7007` | `0.120699` |
| KNN (`k = 13`) | `0.6532` | `0.6459` | `6.9170` | `9.41` | `77.9367` | `0.120111` |
| Gradient Boosting (default) | `0.8503` | `0.7107` | `6.3128` | `8.59` | `63.6774` | `0.108569` |
| Gradient Boosting (tuned) | `0.7878` | `0.7223` | `6.1340` | `8.35` | `61.1212` | `0.106367` |

### Raw optimization and comparison outputs preserved from the notebook

- Initial linear-regression cost with `w = 0` and `b = 0`: `1205.7893438889628`
- Linear regression cost progression:
  - `Iteration    0: Cost  1180.48`
  - `Iteration  900: Cost    46.46`
- Polynomial regression cost progression:
  - `Iteration    0: Cost   800.96`
  - `Iteration  900: Cost    30.97`
- Regularized polynomial regression cost progression:
  - `Iteration    0: Cost   800.97`
  - `Iteration  900: Cost    37.73`
- Regularized linear regression cost progression:
  - `Iteration    0: Cost  1180.48`
  - `Iteration  900: Cost    48.01`
- RFE-selected regularized linear regression cost progression:
  - `Iteration    0: Cost  1181.15`
  - `Iteration  900: Cost    49.72`
- RFE-selected regularized polynomial regression cost progression:
  - `Iteration    0: Cost   936.08`
  - `Iteration  900: Cost    47.53`
- Best KNN output:
  - `Highest R2 score on the test set: 0.6459 at k = 13`
  - `R2 score on train set: 0.6532`
- Gradient Boosting grid search output:
  - `Fitting 5 folds for each of 312 candidates, totalling 1560 fits`
  - `Best parameters: {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300}`
  - `Best cross-validation score: 0.6185460330268293`

### Representative prediction outputs printed by the notebook

- Linear Regression:
  - `43.30` vs `29.4`
  - `36.44` vs `28.7`
  - `55.74` vs `33.95`
  - `46.37` vs `40.075`
  - `65.82` vs `56.525`
- Polynomial Regression:
  - `44.24` vs `29.4`
  - `38.88` vs `28.7`
  - `57.75` vs `33.95`
  - `43.46` vs `40.075`
  - `67.48` vs `56.525`
- Polynomial Regression Regulated:
  - `39.95` vs `29.4`
  - `37.14` vs `28.7`
  - `54.12` vs `33.95`
  - `41.03` vs `40.075`
  - `62.83` vs `56.525`
- Linear Regression Regulated:
  - `43.28` vs `38.5`
  - `35.54` vs `31.15`
  - `36.99` vs `33.95`
  - `70.87` vs `79.1`
  - `35.61` vs `40.6`
- Linear Regression RFE selection:
  - `41.19` vs `38.5`
  - `35.58` vs `31.15`
  - `37.83` vs `33.95`
  - `71.31` vs `79.1`
  - `35.38` vs `40.6`
- Polynomial Regression RFE selection:
  - `38.07` vs `38.5`
  - `31.86` vs `31.15`
  - `41.93` vs `33.95`
  - `80.57` vs `79.1`
  - `34.08` vs `40.6`

### Visual diagnostics produced in the notebook

The notebook includes:

- 14 scatter plots of each feature against price
- original-vs-normalized histograms for `area`, `parking`, and `mainroad`
- cost-reduction curves for every custom gradient-descent-trained model
- residual histograms with KDE for training and testing predictions for each major model family
- bar-chart comparisons across `R^2`, `MAE`, average percentage error, `MSE`, and `NRMSE`
- a `k`-versus-`R^2` plot for KNN model selection
- a boosting-iteration-versus-training-error curve for Gradient Boosting

## Findings and Observations

This section consolidates the explicit observations stated in the notebook, together with direct conclusions that follow from the printed outputs.

### Explicit notebook observations and comments

- The notebook describes Z-score normalization as a method that makes the mean `0` and the standard deviation `1`, placing features on a common scale.
- It states that the initial cost at zero weights is high and should be improved by gradient descent.
- It explains KDE as a smoothed estimate of the residual distribution.
- It notes that NRMSE is useful because it normalizes prediction error by the range of the target variable.
- It explicitly tests regularization on the original linear model to see whether metrics improve.
- It explains RFE as an iterative procedure that removes less important features based on model coefficients.

### Data-quality and preprocessing findings directly shown in outputs

- Outlier removal reduces the dataset from `545` rows to `530` rows.
- The target column `price` is originally `int64`.
- After preprocessing, the notebook reports `Columns containing strings: []`.
- The notebook prints `No null values`.
- The notebook prints `No duplicates`.
- The train/test split is `371/159`.

### Modeling findings directly supported by the reported metrics

- Plain linear regression generalizes better than unregularized polynomial regression on the held-out test set.
- The unregularized degree-2 polynomial model achieves a higher training `R^2` (`0.7712`) but a substantially lower testing `R^2` (`0.5097`), which is consistent with overfitting.
- Adding L2 regularization improves the polynomial model's test performance from `0.5097` to `0.6272`.
- Regularizing the original linear model improves test performance from `0.6824` to `0.6939` and reduces both `MAE` and `MSE`.
- RFE with 9 selected features does not outperform the full regularized linear model on the test set.
- Polynomial expansion after RFE also does not outperform the full regularized linear model.
- KNN reaches its best test `R^2` at `k = 13`.
- Default Gradient Boosting outperforms all of the custom regression variants on held-out `R^2`.
- Tuned Gradient Boosting is the best-performing model in the notebook on test metrics.

### Best-performing configurations

- Best from-scratch model:
  - Linear Regression Regulated
  - Test `R^2 = 0.6939`
  - `MAE = 6.2286`
  - `MSE = 67.3720`
  - `NRMSE = 0.111674`
- Best overall model in the notebook:
  - Gradient Boosting with adjusted hyperparameters
  - Test `R^2 = 0.7223`
  - `MAE = 6.1340`
  - `MSE = 61.1212`
  - `NRMSE = 0.106367`


## Mathematical Foundations Demonstrated

- multivariate linear regression formulation
- batch gradient descent
- mean squared error objective construction
- partial derivatives with respect to weights and bias
- L2 regularization and its effect on the gradient
- vector and matrix operations with NumPy
- dot-product-based prediction
- feature scaling through Z-score normalization
- polynomial basis expansion
- feature selection via coefficient-based elimination
- residual analysis and density estimation
- train/test generalization analysis
- hyperparameter search and model selection

## Tech Stack

- Python
- NumPy
- Jupyter Notebook / Google Colab
- Kaggle dataset (`Housing.csv`)
- pandas
- matplotlib
- seaborn
- scikit-learn

