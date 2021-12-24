const tf = require('@tensorflow/tfjs')

function train(x, y, epcs) {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    model.compile({
        optimizer: tf.train.sgd(0.01),
        loss: "meanSquaredError",
        metrics: [tf.metrics.meanAbsoluteError],
    });
    model.fit(x, y, {
        // batchSize: 32,
        epochs: 100,
        shuffle: true,
        validationSplit: 0.1
    });
    return model;
}


const xs = tf.tensor1d([...Array(20)].map((_, i) => i));
const ys = tf.tensor1d([...Array(20)].map((_, i) => i * 2));
xs.print();
ys.print();
const model = train(xs, ys);
y_pred = model.predict(tf.tensor1d([4]));
y_pred.print();
