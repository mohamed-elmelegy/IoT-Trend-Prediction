#! /usr/bin/env node
/**
 *
 */
// "use strict";

const tf = require("@tensorflow/tfjs");
const { AR, ARIMA } = require("./arima");
const dfd = require("danfojs-node");

let params = {
    epochs: 1024,
    batchSize: 10,
    validationSplit: .2,
    shuffle: false,
    verbose: 0
};


var arima = ARIMA(2, 0, 2);
dfd.read_csv("../../../Data/test.csv").then(df => {
    df.print();
    return df["value"].tensor.array();
}).catch(console.error);
// .then(X => {
//     return arima.fit(X, params);
// }).then(() => {
//     console.log(arima.p, arima.d, arima.q);
//     return arima.predict(Array(60));
// }).then(console.log).catch(console.error);


// let X = tf.linspace(1, 50, 50).arraySync();
// let arima = ARIMA(2, 0, 2);
// let arima = AR(2);
// let params = {
//     epochs: 1024,
//     batchSize: 10,
//     validationSplit: .2,
//     shuffle: false,
//     verbose: 0
// };
// arima.fit(X, params).then(() => {
//     console.log(
//         // "Final ARIMA(" + arima.p + ", " + arima.d + ", " + arima.q + ") Predictions: "
//         `Final ARIMA(${arima.p}, ${arima.d}, ${arima.q}) Predictions:`
//     );
//     return arima.predict(Array(5));
// }).then(console.log).catch(err =>
//     console.error(err)
// );