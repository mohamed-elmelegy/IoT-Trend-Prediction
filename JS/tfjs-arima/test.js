#! /usr/bin/env node
/**
 *
 */
// "use strict";




const tf = require("@tensorflow/tfjs");
const { AR, ARIMA } = require("./arima");


let X = tf.linspace(1, 50, 50).arraySync();
let arima = ARIMA(2, 0, 2);
// let arima = AR(2);
let params = {
    epochs: 1024,
    batchSize: 10,
    validationSplit: .2,
    shuffle: false,
    verbose: 0
};
// arima.fit -> arima.fitSync -> arima.arModel.fit.then.then(res => {})
arima.fit(X, params).then(() => {
    console.log(
        // "Final ARIMA(" + arima.p + ", " + arima.d + ", " + arima.q + ") Predictions: "
        `Final ARIMA(${arima.p}, ${arima.d}, ${arima.q}) Predictions:`
    );
    return arima.predict(Array(5));
}).then(console.log).catch(err =>
    console.error(err)
);