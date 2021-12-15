/* ============================ Simple ARIMA ============================ */

// Load package
const ARIMA = require('arima')

// Synthesize timeseries
var ts = Array(12*10).fill(0).map((_, i) => i + Math.random() / 5)

// Init arima and start training
const arima = new ARIMA({
  p: 2,
  d: 1,
  q: 2,
  verbose: false
}).train(ts)

// Predict next 12 values
const arima_forecast = arima.predict(12)


/* ============================ Seasonal ARIMA ============================ */

// Init sarima and start training
const sarima = new ARIMA({
    p: 2,
    d: 1,
    q: 2,
    P: 1,
    D: 0,
    Q: 1,
    s: 12,
    verbose: false
  }).train(ts)
  
  // Predict next 12 values
  const sarima_forecast = sarima.predict(12)


/* =============== Seasonal ARIMA Using Exogenous Variables =============== */

// Generate timeseries using exogenous variables
const f = (a, b) => a * 2 + b * 5
const exog = Array(30).fill(0).map(x => [Math.random(), Math.random()])
const exognew = Array(10).fill(0).map(x => [Math.random(), Math.random()])
var ts = exog.map(x => f(x[0], x[1]) + Math.random() / 5)

// Init and fit sarimax
const sarimax = new ARIMA({
  p: 1,
  d: 0,
  q: 1,
  transpose: true,
  verbose: false
}).fit(ts, exog)

// Predict next 12 values using exognew
const sarimax_forecast = sarimax.predict(12, exognew)


/* ============================ Auto ARIMA ============================ */

// fit time series to auto ARIMA model
const autoarima = new ARIMA({ auto: true }).fit(ts)

// Predict next 12 values
const auto_arima_forecast = autoarima.predict(12)

console.log(auto_arima_forecast)