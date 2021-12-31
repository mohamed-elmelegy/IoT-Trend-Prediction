#! /usr/bin/env node
/**
 *
 */
// "use strict";




const tf = require("@tensorflow/tfjs");
const { AR, MA, ARMA, ARIMA } = require("./arima");


let X = tf.linspace(1, 20, 20).arraySync();
let train = X.slice(0, -5);
let arima = ARIMA(1, 0, 1);
let params = {
    epochs: 100,
    batchSize: 10,
    validationSplit: .2,
    shuffle: false,
    verbose: 0
};
arima.fit(train, params).then(() => {
    return arima.predict(Array(5));
}).then((preds) => {
    console.log(
        "Actual Data: ", 
        X.slice(-5),
        `\n\nFinal ARIMA(${arima.p}, ${arima.d}, ${arima.q}) Predictions:`, 
        preds
    );
    console.log(
        "MSE Score: ", 
        arima.evaluate(X.slice(-5), preds, tf.metrics.meanSquaredError).mean().arraySync()
    );
}).catch(err =>
    console.error(err)
);