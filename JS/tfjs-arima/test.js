#! /usr/bin/env node
/**
 *
 */
// "use strict";

const tf = require("@tensorflow/tfjs");
const { AR, ARIMA } = require("./arima");
// const dfd = require("danfojs-node");

// let params = {
//     epochs: 100,
//     batchSize: 10,
//     validationSplit: .2,
//     shuffle: false,
//     verbose: 0
// };

// console.log(process.env.PWD);

// var arima = ARIMA(2, 0, 2);
// dfd.read_csv("./Data/daily-total-female-births-in-cal.csv").then(df => {
//     df.print();
//     return df["births"].tensor.array();
// }).then(X => {
//     console.log(X.slice(-15));
//     X = X.slice(0, -15)
//     return arima.fit(X, params);
// }).then(() => {
//     console.log(arima.p, arima.d, arima.q);
//     return arima.predict(Array(15));
// }).then(console.log).catch(console.error);

const X = tf.randomNormal([1000], 50, 5).arraySync();
const trainSize = parseInt(0.8 * X.length);
const xTrain = X.slice(0, trainSize);
const xTest = X.slice(-1 * (X.length - trainSize));
// console.log(xTrain, xTest);

let params = {
    epochs: 2500,
    batchSize: parseInt(0.2 * xTrain.length),
    validationSplit: .2,
    callbacks: tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 2 }),
    shuffle: false,
    verbose: 1
};

let arima = ARIMA(4, 0, 1);

console.time("Total Fit Time");

arima.fit(X, params).then(() => {
    console.log(`ARIMA(${arima.p}, ${arima.d}, ${arima.q}) Summary:\n----------------------`);
    console.timeEnd("Total Fit Time");

    return arima.predict(xTest.length);
}).then(preds => {
    console.time("Total Predict Time");
    console.log(
        "\nFirst Predictions:",
        preds.slice(0, 5)
    );
    console.timeEnd("Total Predict Time");
    console.log(
        "\nPrediction MSE Loss: ",
        arima.evaluate(xTest, preds, tf.metrics.meanSquaredError).mean().arraySync()
    );
}).catch(err =>
    console.error(err)
);
