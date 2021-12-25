const { model, models } = require("@tensorflow/tfjs");
const tf = require("@tensorflow/tfjs");
const ar = require("./ar");
const ma = require("./ma");

function AR(p = 0, data) {
    const dataHandle = new Promise((resolve, reject) => {
        let [arr_X, arr_y] = ar.pShift(p, data);
        let X = tf.tensor(arr_X).transpose();
        let y = tf.tensor(arr_y).reshape([-1,1]);
        modelParams = {
            "X": X,
            "y": y,
            "inputShape": [X.shape[1]],
            "optimizer": tf.train.adam(0.01),
            "loss": "meanSquaredError",
            "metrics": [tf.metrics.meanSquaredError],
            "epochs": 100, 
            "batchSize": 2, 
            "validationSplit": 0.15
        }
        resolve(modelParams);
    });

    dataHandle.then(params => {
        let modelHandle = new Promise((resolve, reject) => {
            let model = ar.buildModel(params.inputShape, params.optimizer, params.loss, params.metrics);
            resolve(model);
        });

        modelHandle.then(model => {
            trainHandle = new Promise((resolve, reject) => {
                modelFit = ar.fit(model, params.X, params.y, params.epochs, params.batchSize, params.validationSplit);
                resolve(modelFit);
            });

            trainHandle.then(modelFit => {
                console.log(modelFit.history.loss[0]);
                modelPred = ar.predict(model, params.X, 1);
                console.log("Predit One Single Step: ", modelPred.dataSync());
            });
        });
    });

/* 
    TODO: 
    To Be Cont'd In AR:
    -------------------
    1) Predict with input steps not just the next one only
    2) Change it all to Object Oriented Style
    3) Change it to be more asynchronous
    4) Follow JS coding standards, Document all functions, leave more comments
 */

}

// Test Auto Regression
AR(2, tf.linspace(1, 250, 500).dataSync());