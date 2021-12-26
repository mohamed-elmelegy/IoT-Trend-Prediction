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
            // Add temporary 0 instead of NAN... TODO: Input Data should be cleaned
            shiftedX.push(0);
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
    // Define input, which has a size of inputShape
    const inputLayer = tf.input({ shape: inputShape });

    // Output dense layer uses linear activation.
    const denseLayer1 = tf.layers.dense({ units: 1 });

    // Obtain the output symbolic tensor by applying the layers on the inputLayer.
    const output = denseLayer1.apply(inputLayer);

    // Create the model based on the inputs.
    const model = tf.model({ inputs: inputLayer, outputs: output });

    model.compile({
        optimizer: optimizer,
        loss: loss,
        metrics: metrics,
    });
    model.summary();

    return model
}

async function fit(model, X, y, epochs = 100, batchSize = 32, validationSplit = 0.20) {
    const modelFit = await model.fit(X, y, {
        batchSize: batchSize,
        epochs: epochs,
        shuffle: false,
        validationSplit: validationSplit
    });
    return modelFit;
}

function predict(model, labels, p, steps = 1) {
    res = labels.slice(-p);
    const shape = [1, p];
    for (let s = 0; s < steps; s++) {
        features = tf.tensor(res.slice(-p)).reshape(shape);
        yHat = model.predict(features).arraySync();
        res.push(yHat[0]);
    }
    return res.slice(-steps);
}

function evaluate(yTrue, yPred, fn) {
    return fn(yTrue, yPred);
}

module.exports = {
    pShift,
    buildModel,
    fit,
    predict,
    evaluate
}