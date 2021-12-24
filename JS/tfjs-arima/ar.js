const tf = require("@tensorflow/tfjs");

function pShift(p, X) {
    if (p <= 0) {
        return [X, X]
    }
    let shift = 1;
    let shiftedXs = [];
    let inputArray = [...X];
    let outputArray = [...X];
    while (shift <= p) {
        let shiftedX = [];
        for (let k = 0; k < shift; k++) {
            shiftedX.push(NaN);
        }
        for (let i = 0; i < outputArray.length - shift; i++) {
            shiftedX.push(X[i]);
        }
        outputArray = shiftedX;
        shiftedXs.push(shiftedX);
        // inputArray.shift();
        shift += 1;
    }
    // console.log(X, "\n------------------\n", inputArray, "\n------------------\n", shiftedXs);

    return [shiftedXs, inputArray];
}

function buildModel(inputShape, optimizer, loss, metrics) {
    console.log(inputShape);
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [3, 100] }));
    model.compile({
        optimizer: optimizer,
        loss: loss,
        metrics: metrics,
    });

    return model
}

function fit(model, X, y, epochs = 100, batchSize = 32, validationSplit = 0.20) {
    X = tf.tensor(X);
    y = tf.tensor(y);
    model.fit(X, y, {
        batchSize: batchSize,
        epochs: 100,
        shuffle: true,
        validationSplit: validationSplit
    });

    return model;
}

function predict(model, X) {
    return model.predict(X);
}

function evaluate(yTrue, yPred, fn) {
    return true
}

module.exports = {
    pShift,
    buildModel,
    fit,
    predict,
    evaluate
}