#! /usr/bin/env node
/**
 *
 */
// "use strict";

const tf = require("@tensorflow/tfjs");
const { AR, ARIMA } = require("./arima");
const dfd = require("danfojs-node");

let params = {
    epochs: 100,
    batchSize: 10,
    validationSplit: .2,
    shuffle: false,
    verbose: 0
};

console.log(process.env.PWD);

var arima = ARIMA(2, 0, 2);
dfd.read_csv("./Data/daily-total-female-births-in-cal.csv").then(df => {
    df.print();
    return df["births"].tensor.array();
}).then(X => {
    console.log(X.slice(-15));
    X = X.slice(0, -15)
    return arima.fit(X, params);
}).then(() => {
    console.log(arima.p, arima.d, arima.q);
    return arima.predict(Array(15));
}).then(console.log).catch(console.error);


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
