#! /usr/bin/env node
/**
 *
 */
// "use strict";

const tf = require("@tensorflow/tfjs");
const { AR, ARIMA } = require("./arima");

let X = tf.linspace(1, 50, 50).arraySync();
let arima = ARIMA(2, 0, 2);
let params = {
    epochs: 500,
    batchSize: 10,
    validationSplit: .2,
    shuffle: false
};
arima.fit(X, params).then(() => {
    console.log("Final ARMA Predictions: ", arima.predictSync(Array(5)));
}).catch(err =>
    console.error(err)
);