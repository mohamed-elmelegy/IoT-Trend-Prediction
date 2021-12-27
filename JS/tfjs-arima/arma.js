// const { model, models, step } = require("@tensorflow/tfjs");
const tf = require("@tensorflow/tfjs");
const ar = require("./ar");
const ma = require("./ma");

function AR(p = 0, data) {
	dataHandle().then(params => {
		return [params, ar.buildModel(params.inputShape,
			params.optimizer,
			params.loss,
			params.metrics
		)];
	}).then(res => {
		[params, mod] = res;
		// console.time("fit");
		ar.fit(mod,
			params.X,
			params.y,
			params.epochs,
			params.batchSize,
			params.validationSplit
		).then(modelFit => {
			// console.timeEnd("fit");
			steps = 5;
			modelPred = ar.predict(mod, data, p, steps);
			mse = ar.evaluate(tf.linspace(251, 255, 5),
				tf.tensor(modelPred),
				tf.metrics.meanSquaredError
			);
			console.log("Training Loss: ", modelFit.history.loss.slice(-1));
			console.log("Predict (" + steps + ") Step: ", modelPred);
			console.log("MSE Metric: ", mse.mean().dataSync());
		}).catch(err => {
			console.error(err);
		});
	});// FIXME might need a finally clause
	// TODO: fit() & predict() must be async
	// TODO: fitSync() & predictSync() anything needed here must be sync

	function dataHandleSync() {
		let [arr_X, arr_y] = ar.pShift(p, data);
		let X = tf.tensor(arr_X).transpose();
		let y = tf.tensor(arr_y).reshape([-1, 1]);
		modelParams = {
			"X": X,
			"y": y,
			"inputShape": [X.shape[1]],
			"optimizer": tf.train.adam(0.01),
			"loss": "meanSquaredError",
			"metrics": [tf.metrics.meanSquaredError],
			"epochs": 1024,
			"batchSize": 10,
			"validationSplit": 0.10
		};
		return modelParams;
	}

	function dataHandle() {
		return Promise.resolve(dataHandleSync());
	}

	/* 
			TODO: 
			To Be Cont'd In AR:
			-------------------
			1) Predict with input steps not just the next one only
			2) Using Danfo.js in any preprocessing
			3) Change it all to Object Oriented Style
			4) Change it to be more asynchronous
			5) Follow JS coding standards, Document all functions, leave more comments
	 */

}

// Test Auto Regression
AR(2, tf.linspace(1, 50, 50).arraySync());