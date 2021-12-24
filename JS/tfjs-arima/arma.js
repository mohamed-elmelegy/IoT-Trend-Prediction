const { model } = require("@tensorflow/tfjs");
const tf = require("@tensorflow/tfjs");
const ar = require("./ar");
const ma = require("./ma");

function AR(p = 0, data) {
    let [X, y] = ar.pShift(p, data);
    const inputShape = [X.length];
    const optimizer = tf.train.sgd(0.01);
    const loss = "meanSquaredError";
    const metrics = [tf.metrics.meanSquaredError];
    let model = ar.buildModel(inputShape, optimizer, loss, metrics);
    model = ar.fit(model, X, y, epochs = 100, batchSize = 32, validationSplit = 0.20);
    let yPred = ar.predict(model, X);

    return yPred;
}

// Test Auto Regression
AR(3, [...Array(100).keys()]);