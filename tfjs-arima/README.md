# Non-Seasonal ARIMA for real time trend prediction on [MoT](http://www.masterofthings.com/)

A [TensorFlow JavaScript](https://www.tensorflow.org/js) approach to implement ARIMA (AutoRegression Integrated Moving Average) model

<!-- > This document serves as the manual for the module -->

This module implements the following models from ARIMA family:

- AR(p).
- MA(q).
- ARMA(p,q).
- ARIMA(p, d, q)

## Files


### `arima.js`

Provides ARIMA model APIs. The full equation for the non-seasonal arima model is

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
const tf = require("@tensorflow/tfjs");

var data = [...];

var model = ARIMA(p, d, q);

let params = {
    epochs: 100,
    batchSize: 32,
    validationSplit: .2,
    callbacks: tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 3 }),
    shuffle: false
};

model
    .fit(data, params)
    .then(() => {
        let steps = 5;
        return arima.predict(steps);
    })
  .catch(console.error)
  .finally(event.end);
```

`ARIMA(p, d, q)`

- `p`: order of the AR `auto regression` terms.
- `d`: order of differencing.
- `q`: order of the MA `moving average` terms.

> `ARIMA` order of (`p`, `d`, `q`) should be provided in full. (`1`, `0`, `0`) is pure `AR`, (`0`, `0`, `1`) is pure `MA`, (`1`, `0`, `1`) is `ARMA`.

---

`async fit(X, params)`

- `X`: the time series to fit against.
- `params` (object): This object includes all available hyperparameters and on-training events which provided in TensorFlow.js by default, So for more details about available keys please check `args` fields in `fit()` method parameters in [TensorFlow.js Docs](https://js.tensorflow.org/api/latest/#tf.LayersModel.fit). Here are some of them:
  - `epochs`: (number) Integer number of times to iterate over the training data arrays.
  - `batchSize`: (number) Number of samples per gradient update. If unspecified, it will default to $32$.
  - `callbacks`: List of callback functions to be called during training. For example:
    - `earlyStopping` (object): Factory function for a Callback that stops training when a monitored quantity has stopped improving.
    Early stopping is a type of regularization, and protects model against overfitting.
      - `args` (Object): Optional
        - `monitor` (string): Quantity to be monitored. Defaults to **val_loss**.
        - `minDelta` (number): Minimum change in the monitored quantity to qualify as improvement, i.e., an absolute change of less than minDelta will count as no improvement. Defaults to $0$.
        - `patience` (number): Number of epochs with no improvement after which training will be stopped. Defaults to $0$.
    - `onTrainBegin` (logs): called when training starts.
    - `onTrainEnd` (logs): called when training ends.
    - `onEpochBegin` (epoch, logs): called at the start of every epoch.
    - `onEpochEnd` (epoch, logs): called at the end of every epoch.
  - `shuffle` (boolean): Whether to shuffle the training data before each epoch. Has no effect when stepsPerEpoch is not null. It is necessary to be $false$ with time series data.
  - `validationSplit` (number): Float between 0 and 1: fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch.
  - `verbose`: Verbosity level. Expected to be $0$ or $1$. Default: $1$.<br>
    0 - No printed message during fit() call. <br>
    1 - In Node.js (tfjs-node), prints the progress bar, together with real-time updates of loss and metric values and training speed.
<br>

Returns `{Promise<tf.History>}`

---

`async predict(stepsNumber)`

- `stepsNumber`: number of future steps to forecast.<br>

Returns `{Promise<Array>}`

---

`async evaluate(yTrue, yPredicted, fn?)`

- `yTrue` (Array): the actual correct values of predicted steps. Length must be the same for `yPredicted`<br>
- `yPredicted` (Array): the predicted of forecasted values with some steps. Length must be the for `yTrue`<br>
- `fn` (Function): Optional param to use in the evaluation of model behaviour. Defaults to `tf.metrics.meanSquaredError`. <br>

Returns `{Promise<tf.Tensor>|Number|Array}`

---

## Full Example

This example includes:
- Simple Time Series Train-Test Split.
- Training ARIMA(2, 2, 2) model using several params.
- How long it takes to finish ARIMA training?
- How many epochs it takes from the given $2000$ to finish training?
- Predicting number of upcoming steps equals to `xTest` length.
- How long it takes to predict number of upcoming steps equals to `xTest` length?
- Evaluation of ARIMA model using `MeanSquaredError (MSE)` metric.

### Code:

```js
const tf = require("@tensorflow/tfjs");
const { ARIMA } = require("./arima");

// 1-D Time Series Dataset
const X = [...];

// Train-Test Split 
const trainSize = parseInt(0.8 * X.length);
const xTrain = X.slice(0, trainSize);
const xTest = X.slice(-1 * (X.length - trainSize));

// ARIMA Model Params
let params = {
    epochs: 2000,
    batchSize: parseInt(0.2 * xTrain.length),
    validationSplit: .2,
    callbacks: tf.callbacks.earlyStopping({ 
      monitor: 'val_loss', 
      patience: 3 
    }),
    shuffle: false,
    verbose: 1
};

// ARIMA model instance
let arima = ARIMA(2, 2, 2);

console.time("Total Fit Time");

// Start ARIMA training
arima.fit(xTrain, params).then((modelFit) => {
    console.log(`ARIMA(${arima.p}, ${arima.d}, ${arima.q}) Summary:\n----------------------`);
    console.timeEnd("Total Fit Time");
    console.log("AR stopped training after just: #", modelFit.history.loss.length, "epochs");
    
    // Start ARIMA forecasting
    return arima.predict(xTest.length);
}).then(preds => {
    console.time("Total Predict Time");
    console.log(
        "\nActual Test Data Sample:",
        xTest.slice(0, 5),
        "\nPredictions Sample:",
        preds.slice(0, 5)
    );
    console.timeEnd("Total Predict Time");
    
    // Start ARIMA evaluation
    console.log(
        "\nPrediction MSE Loss: ",
        arima.evaluate(xTest, preds, tf.metrics.meanSquaredError).mean().arraySync()
    );
}).catch(err =>
    console.error(err)
);
```

### Results Summary:

> The results of this example are obtained from testing the model on a local environment and [this time series dataset](./data.txt) with size 365 records.

```
  ARIMA(2, 2, 2) Summary:
  ----------------------
  Total Fit Time: 485.274ms
  AR stopped training after just: (4) epochs

  Actual Test Data Sample Of (5): [ 
    42, 
    38, 
    47, 
    38, 
    36 
  ]
  Predictions Sample Of The Same (5): [
    44.83704291284084,
    42.89202205091715,
    44.843544685747474,
    42.89952243026346,
    44.85235397005454
  ]
  Total Predict Time: 4.004ms

  Prediction MSE Loss:  42.57068634033203
```


