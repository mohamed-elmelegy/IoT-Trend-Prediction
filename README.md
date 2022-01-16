# Non-Seasonal ARIMA for real time trend prediction on [MoT](http://www.masterofthings.com/)

A pure JavaScript approach to implement ARIMA (AutoRegression Integrated Moving Average) model

<!-- > This document serves as the manual for the module -->

This module implements the following models from ARIMA family:

- AR(p).
- MA(q).
- ARMA(p,q).
- ARIMA(p, d, q)

## Files

### `numpy.js`

provides vector math operations built on top of JavaScript's `Array` in a familiar syntax to Python's [numpy](https://pypi.org/project/numpy/)

> The module only implements required operations for ARIMA, supporting up to only 2D operations

Once you have your data into an array, you can convert it to an `NDArray` like this:

```js
const np = require("./numpy");

var data = [...];
var vector = np.array(data);
```

<!-- ### `linreg.js`

Provides linear regression for using inside ARIMA model.

Usage:

```js
const { GradientDescent } = require("./linreg");

var model = new GradientDescent(learningRate, options);

model
  .fit(features, labels, maxIter, stopThreshold)
  .then((fit) => {
    var fitWeights = fit._coef;
    return fit.predict(testFeatures);
  })
  .then((predictions) => {
    var mean = predictions.mean();
    var stdDeviation = predictions.std();
    var error = testLabels.sub(predictions);
    var mae = error.apply(Math.abs).mean(); // mean absolute error
    var mse = error.power(2).mean();
  })
  .catch(console.error)
  .finally(event.end);
```

`GradientDescent(learningRate, options)`

- `learningRate` is the hyperparameter &alpha; used with gradient descent algorithm. Defaults to $10^{-3}$.
- `options` is an object with possible fields:
  - `weights`: optional initial weights of parameters. Defaults to random weights.
  - `momentum`: optional hyperparameter &gamma; used for Nesterov update in gradient descent algorithm. Defaults to $0$.
  - `batchSize`: optional batch size to alter the gradient descent variant. `0|null` is vanilla gradient descent, $1$ is stochastic gradient descent, any `number` value below the data length is mini-batch gradient descent. Defaults to `null`
  - `costFunction`: the cost function the gradient descent should use to minimise. Defaults to mean squared error.
  - `gradient`: callable to calculate gradient according to the `costFunction`. The arguments should be `featureVector`, `actualObservations`, `predictedObservations`. Defaults to the gradient of the mean square error.
  - `nesterov`: a flag to use Nesterov update. Defaults to `false`.

---

`async fit(features, labels, maxIter, stopThreshold)`:

- `features`: the feature vector holding the independent features.
- `labels`: the labels vector holding the dependent feature. This must be of the same length as that of `features`
- `maxIter`: maximum number of epochs after which the algorithm must stop. Defaults to $1024$.
- `stopThreshold`: the minimum difference in cost or norm 2 of gradient, below which the algorithm should terminate. Defaults to $10^{-6}$.

---

`async predict(features)`

- `features`: feature vector of independent variables.

--- -->

### `utils.js`

Provides utilities to convert from MoT's default Array of objects (or records) into a `tensorflow` tensor.

Usage:

```js
const tf = require("@tensorflow/tfjs-node");
const utils = require("./utils");

var tensorX = tf.tensor(...);
var ord = 0;
var tensorXDiff = utils.diff(tensorX, ord);

var data = [{...}, ..., {...}]; // records from MoT
var [index, vector] = utils.extractIdxTensor1D(data);
// index instanceof Array; vector instanceof tf.Tensor
var vector = utils.extractTensor2D(data);
// vector instanceof tf.Tensor
```

- `diff`: applies the differencing operation $D_i=V_{i+1}-V_i; i\in[1,N]$. The resultant tensor's length is shorter by `ord` than original tensor length. `ord` defaults to `1`.
- `extractIdxTensor1D`: extracts index from `data` into `Array`, and values of data into `tf.Tensor`. **Assumption**: any object in `data` has `TimeStamp` for index, and a single column with arbitrary name which contains the value.
- `extractTensor2D`: same as `extractIdxTensor1D`, but the result is a single `tf.Tensor` that holds indices at the first column, and values at the second column.

---

### `arima.js`

Provides the ARIMA model APIs. The full equation for the non-seasonal arima model is

$$
\Phi(B)(y_t'-\mu)=\Theta(B)\varepsilon_t
$$

where

- $B$ is the backshift operator: $By_t=y_{t-1};B\varepsilon_t=\varepsilon_{t-1};B\phi=\phi;B^dy_t=y_{t-d}$.
- $y_t'$ is the differenced series: $y_t'=y_t-y_{t-1}$
- $\mu$ is the mean of the differenced series: $\mu=\frac{1}{T}\sum_{t=0}^Ty_t'$.
- $\varepsilon_t$ is the residual: $\varepsilon_t=y_t-\hat{y_t}$.
- $\Phi$ are the auto regression weights: $\Phi(B)=(1-B\phi_1-...-B^p\phi_p)$.
- $\Theta$ are the moving average weights: $\Theta(B)=(1+B\theta_t+...+B^q\theta_q)$.

Usage:

```js
const { ARIMA } = require("./arima");

var vector = [...];
var order = [p, d, q];
var model = ARIMA(order, options);
model
  .fit(vector, maxIter, stopThreshold)
  .then((fit) => {
    var metrics = fit.metrics;
    var phi = fit.phi;
    var theta = fit.theta;
    var intercept = fit.intercept;
    return fit.forecast(periods);
  })
  .catch(console.error)
  .finally(event.end);
```

`ARIMA([p, d, q], options)`

- `p`: order of the AR `auto regression` terms.
- `d`: order of differencing.
- `q`: order of the MA `moving average` terms.
- `options`: hyperparameters for the implementation of the model:
  - `learningRate`: the hyperparameter &alpha; for the underlying gradient descent used. Defaults to $10^{-1}$.
  - `batchSize`: the number of records to used inside the gradient descent. Use `null|0` for batch/vanilla gradient descent, $1$ for stochastic gradient descent, or any `number` less than the observations number for mini-batch gradient descent.

> The order array should be provided in full. [1, 0, 0] is pure AR, [0, 0, 1] is pure MA, [1, 0, 1] is ARMA.

---

`async fit(series, maxIter, stopThreshold)`

- `series`: the time series to fit against.
- `maxIter`: the number of epochs to be used in gradient descent. Defaults to $1024$.
- `stopThreshold`: the threshold below which the gradient descent stops looping into epochs. Defaults to $10^{-6}$.

---

`async forecast(periodsNumber)`

- `periods`: number of future periods to forecast into

---

## License
