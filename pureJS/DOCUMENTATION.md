# Technical documentation for the pure JavaScript non-seasonal ARIMA

## `numpy.js`

```js
array(arrLike);
```

converts an array or an array-like object into an `NDArray` object

Parameters:

- `arrLike {Array}`: array or array-like object

Returns:

- `vector {NDArray}`: the data in the array wrapped inside an `NDArray` object

---

```js
// TODO the remaining numpy utils
```

---

## `utils.js`

```js
diff(tensor, (ord = 1));
```

performs the differencing operation on a `tf.Tensor` object. $T'=T_{i+1}-T_i,i\in[1,N]$

Parameters:

- `tensor {tf.Tensor}`: a tensor object.
- `ord {number}`: an integer representing order of differencing. an order higher than $1$ is calculated by recursively calculating difference of $1$.

Returns:

- `tensor {tf.Tensor}`: the differenced tensor of length `tensor.length - ord`.

Example

```js
const tf = require("@tensorflow/tfjs-node");
const utils = require("./utils");

var tensor = tf.tensor([1, 2, 3, 4, 5]);
var tensorDiff = utils.diff(tensor);
console.log(tensorDiff); // output [1, 1, 1, 1]
tensorDiff = utils.diff(tensorDiff);
console.log(tensorDiff); // output [0, 0, 0]
tensorDiff = utils.diff(tensor, 2);
console.log(tensorDiff); // output [0, 0, 0]
```

---

```js
extractIdxTensor1D(listOfObjects);
```

Extracts the timestamp index and the values from a list of objects. Each object assumed to be composed only of a timestamp, and an arbitrarily named value.

Parameters:

- `listOfObjects {Array}`: array of objects. Each object must have a key of `TimeStamp`, and an arbitrarily named value.

Returns:

- `[index {Array}, value {tf.Tensor}]`: the elements of the objects from the array in an array, the first entry is an array of timestamp indices, the second entry is a tensor with the associated value.

Example:

```js
const utils = require("./utils");

var data = [
  { TimeStamp: 128256512, reading: 64 },
  { TimeStamp: 256512128, reading: 16 },
  { TimeStamp: 512128256, reading: 32 },
];

var [idx, val] = utils.extractIdxTensor1D(data);
console.log(idx); // [128256512, 256512128, 512128256]
val.print(); // Tensor\n\t[64, 16, 32]
```

---

```js
extractTensor2D(listOfObjects);
```

Extracts the index and the value of the list of objects into a single 2D tensor. Each object assumed to be composed only of a timestamp, and an arbitrarily named value.

Parameters:

- `listOfObjects {Array}`: array of objects. Each object must have a key of `TimeStamp`, and an arbitrarily named value.

Returns:

- `tensor {tf.Tensor}`: a 2D tensor where the first column hold the timestamp, and the second contains the associated value.

Example:

```js
const utils = require("./utils");

var data = [
  { TimeStamp: 128256512, reading: 64 },
  { TimeStamp: 256512128, reading: 16 },
  { TimeStamp: 512128256, reading: 32 },
];

var tensor = utils.extractTensor2D(data);
tensor.print(); // Tensor\n\t[[128256512, 256512128, 512128256],\n\t [64, 16, 32]]
```

---

## `linreg.js`

```js
class GradientDescent
```

### Constructor

```js
constructor(learningRate, options);
```

Parameters:

- `learningRate {number}`: the portion of gradient used for updating the weights.
- `options {object}`:
  - `weights {Array}`: initial weights to be used. Defaults to random weights.
  - `momentum {number}`: the value of momentum at which the adaptive learning rate changes. Defaults to $0$.
  - `batchSize {number}`: the number of records used in gradient descent algorithm. `0|null` for vanilla/batch gradient descent. $1$ for stochastic gradient descent. any other number less than the records number for mini-batch gradient descent.
  - `costFunction {function}`: the method to calculate cost. should accept 3 arguments, `labels` for real labels, `predictions` for predicted labels, `divisor` for normalising purposes. Defaults to mean squared error function.
  - `gradient {function}`: method to calculate gradient. should accept 3 arguments, `featureVector` the vector of features, `labels` for real labels, `predictions` for predicted labels. Defaults to the gradient of mean square error.
  - `nesterov {boolean}`: flag to use the Nesterov update instead. Defaults to `false`.

Returns:

- `GradientDescent` instance.

Example:

```js
const { GradientDescent } = require("./linreg");

var alpha = 1e-3;
var options = {
  batchSize: 1,
};

var model = new GradientDescent(alpha, options);
```

### Methods

```js
fitSync(X, y, (maxIter = 1024), (stopThreshold = 1e-6));
```

The process of fitting the model to the data.

Parameters:

- `X {Array}`: The feature vector. could be a 2D array.
- `y {Array}`: The labels vector. must be a 1D array.
- `maxIter {number}`: maximum number of iterations, after which the algorithm must come to a halt.
- `stopThreshold {number}`: the minimum difference below which the algorithm stops before finishing all `maxIter` epochs.

Example:

```js
var feats = [[...], ..., [...]];
var labels = [...];

model.fitSync(feats, labels, 8192, 1e-9);
```

---

```js
async fit(X, y, maxIter=1024, stopThreshold=1e-6)
```

Asynchronous call to the `fitSync` function.

Parameters:

- `X {Array}`: The feature vector. could be a 2D array.
- `y {Array}`: The labels vector. must be a 1D array.
- `maxIter {number}`: maximum number of iterations, after which the algorithm must come to a halt.
- `stopThreshold {number}`: the minimum difference below which the algorithm stops before finishing all `maxIter` epochs.

Example:

```js
var feats = [[...], ..., [...]];
var labels = [...];

model
  .fit(feats, labels)
  .then(fit => {
  // TODO operations on the fit model
  })
  .catch(console.error)
  .finally(event.end);
```

---

```js
evaluate(X);
```

Evaluates the predicted labels, given the feature vector.

Parameters:

- `X {Array}`: the feature vector.

Returns:

- `predictions {Array}`: predicted labels.

Example:

```js
model.fitSync(trainFeatures)
var testFeatures = [[...], ..., [...]];
var predictions = model.evaluate(testFeatures);
```

---

```js
async predict(X)
```

Asynchronous call to `evaluate`.

Parameters:

- `X {Array}`: the feature vector.

Returns:

- `predictions {Array}`: predicted labels.

Example:

```js
model.fitSync(trainFeatures);
var testFeatures = [[...], ..., [...]];
model
  .predict(testFeatures)
  .then(predictions => {
    // TODO use the predictions
  })
  .catch(console.error)
  .finally(event.end);
```

---

## `arima.js`

```js
ARIMA([p, d, q], options);
```

A function to build an `AutoRegressionIntegratedMovingAverage` model.

Parameters:

- `[p {number}, d {number}, q {number}]`: `p` order of auto regression, `d` order of differencing, `q` order of moving average
- `options {object}`:
  - `batchSize {number}`: the number of records used in gradient descent algorithm. `0|null` for vanilla/batch gradient descent. $1$ for stochastic gradient descent. any other number less than the records number for mini-batch gradient descent.
  - `costFunction {function}`: the method to calculate cost. should accept 3 arguments, `labels` for real labels, `predictions` for predicted labels, `divisor` for normalising purposes. Defaults to mean squared error function.
  - `gradient {function}`: method to calculate gradient. should accept 3 arguments, `featureVector` the vector of features, `labels` for real labels, `predictions` for predicted labels. Defaults to the gradient of mean square error.
  - `momentum {number}`: the value of momentum at which the adaptive learning rate changes. Defaults to $0$.
  - `nesterov {boolean}`: flag to use the Nesterov update instead. Defaults to `false`.
  - `weights {Array}`: initial weights to be used. Defaults to random weights.

Returns:

- `AutoRegressionIntegratedMovingAverage` instance.

Example

```js
const { ARIMA } = require("./arima");

var model = ARIMA([1, 0, 1], { learningRate: 1e-2, batchSize: 1 });
```

---

```js
fitSync(X, (maxIter = 1024), (stopThreshold = 1e-6));
```

The process of fitting the model to the data.

Parameters:

- `X {Array}`: The time series to fit against.
- `maxIter {number}`: the number of epochs to run the fitting process.
- `stopThreshold {number}`: the threshold below which the fitting process concludes early.

Example:

```js
const { ARIMA } = require("./arima");

var series = [...];
var model = ARIMA([0, 1, 1]);
model.fitSync(series, 8192, 1e-5);
```

---

```js
async fit(X, maxIter=1024, stopThreshold=1e-6)
```

The asynchronous call to `fitSync`.

Parameters:

- `X {Array}`: The time series to fit against.
- `maxIter {number}`: the number of epochs to run the fitting process.
- `stopThreshold {number}`: the threshold below which the fitting process concludes early.

Returns:

- `AutoRegressionIntegratedMovingAverage` fit instance.

Example:

```js
const { ARIMA } = require("./arima");

var series = [...];
var model = ARIMA([2, 1, 0]);
model
  .fit(series)
  .then(fit => {
  // TODO operate on the fit model
  })
  .catch(console.error)
  .finally(event.end);
```

---

```js
fitStatSync(X, (maxIter = 1024));
```

A faster fit method, but with reasonably less accurate parameters, exploiting the law of large numbers from statistics.

Parameters:

- `X {Array}`: The series to fit against
- `maxIter {number}`: maximum number of epochs to run the fitting.

Example:

```js
const { ARIMA } = require("./arima");

var series = [...];
var model = ARIMA([1, 2, 1]);
model.fitMLESync(series, 64);
```

---

```js
async fitStat(X, maxIter=1024)
```

Asynchronous call to `fitStatSync` method.

Parameters:

- `X {Array}`: The series to fit against
- `maxIter {number}`: maximum number of epochs to run the fitting.

Returns:

- `AutoRegressionIntegratedMovingAverage` fit instance.

Example:

```js
const { ARIMA } = require("./arima");

var series = [...];
var model = ARIMA([1, 2, 1]);
model
  .fitSync(series, 32).then(fit => {
  // TODO use the fit model
  })
  .catch(console.error)
  .finally(event.end);
```

---

```js
forecastSync(periods);
```

Forecasting `periods` number of periods into the future.

Parameters:

- `periods {number}`: the number of periods to forecast.

Returns:

- `forecastedPeriods {Array}`: array of length `periods` of forecasted values.

Example

```js
model.fitSync(series);
var forecasts = model.forecastSync(10);
```

---

```js
async forecast(periods)
```

Asynchronous call to `forecastSync` method.

Parameters:

- `periods {number}`: the number of periods to forecast.

Returns:

- `forecastedPeriods {Array}`: array of length `periods` of forecasted values.

Example

```js
model.fitSync(series);
model
  .forecast(10)
  .then((forecasts) => {
    // TODO use the forecasted values
  })
  .catch(console.error)
  .finally(event.end);
```

---

```js
updateSync(trueLags);
```

Updating the model's parameters by considering the true values from after the model was fit.

Parameters:

- `trueLags {Array}`: The actual values starting from the end of the series on which the model was last fit.

Example:

```js
var extras = [...];
model.updateSync(extras);
```

---

```js
async update(trueLags)
```

Asynchronous call to `updateSync` method.

Parameters:

- `trueLags {Array}`: The actual values starting from the end of the series on which the model was last fit.

Returns:

- `AutoRegressionIntegratedMovingAverage` updated fit instance.

Example:

```js
var extras = [...];
model
  .update(extras)
  .then(updated => {
    // TODO use the updated fit model
  })
  .catch(console.error)
  .finally(event.end);

```
