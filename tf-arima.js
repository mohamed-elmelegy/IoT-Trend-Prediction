const tf = require("@tensorflow/tfjs");

class LinearRegression {

	/**
	 * @param {number} shifts 
	 * @param {string} modelName 
	 */
	constructor(shifts, modelName = null) {
		if (!(Number.isInteger(shifts) && shifts >= 0)) {
			throw "Error: `shifts` must be 0 or positive Integer";
		}
		this._shifts = shifts;
		this._modelName = modelName;
	}

	get shifts() {
		return this._shifts;
	}

	get modelName() {
		return this._modelName;
	}

	/**
	 * 
	 * @param {tfjs.Tensor|Array} X 
	 * @param {object} params 
	 * 
	 * @returns {Promise<tf.History>}
	 */
	fit(X, params = {}) {
		let [features, labels] = this.shiftInput(X);
		const sliceWindow = -1 * this._shifts;
		this._keptFeatures = X.slice(sliceWindow);

		// FIXME cross-validation might be required
		// TODO for cross validation: https://js.tensorflow.org/api/latest/#tf.LayersModel.fit
		this.buildModel([features.shape[1]],
			tf.train.adam(0.001),
			"meanSquaredError",
			[tf.metrics.meanSquaredError]
		);

		return this.model.fit(features, labels, params);
	}

	/**
	 * 
	 * @param {Array|tfjs.Tensor} series 
	 * @returns 
	 */
	static calculatePeriods(series) {
		// TODO actually calculate periods, or using timestamps
		return series.length;
	}

	/**
	 * 
	 * @param {Array|tfjs.Tensor|number} X 
	 * @param {Boolean} usingFeatures 
	 * 
	 * @returns {Array}
	 */
	predictSync(steps, usingFeatures = false) {
		if (usingFeatures) {
			return this.model.predict(steps).reshape([-1, 1]).arraySync();
		}
		let res = [...this._keptFeatures];
		let featureShape = [1, this._shifts];
		const sliceWindow = -1 * featureShape[1];
		if (this._shifts <= 0) {
			featureShape = [-1];
		}
		
		for (let s = 0; s < steps; s++) {
			let features = tf.tensor(res.slice(sliceWindow)).reshape(featureShape);
			let yHat = this.model.predict(features).arraySync();
			res.push(yHat[0][0]);
		}
		return res.slice(-steps);
	}

	/**
	 * 
	 * @param {Array|tfjs.Tensor|number} X 
	 * @param {Boolean} usingFeatures 
	 * 
	 * @returns {Promise<Array>}
	 */
	predict(X, usingFeatures = false) {
		return Promise.resolve(this.predictSync(X, usingFeatures));
	}

	/**
	 * 
	 * @param {Array} X 
	 * 
	 * @returns {Array<tf.Tensor>}
	 */
	shiftInput(X) {
		if (this._shifts == 0) {
			return [
				tf.tensor(X).reshape([-1, 1]),
				tf.tensor(X).reshape([-1, 1])
			];
		}
		const featureShape = [(X.length - this._shifts), this._shifts];
		const labelShape = [-1, 1];
		let labels = X.slice(this._shifts);
		let features = [];
		for (let i = 1; i <= this._shifts; i++) {
			features.push(X.slice(this._shifts - i, -i));
		}

		return [
			tf.tensor(features).reshape(featureShape),
			tf.tensor(labels).reshape(labelShape)
		];
	}

	/**
	 * TODO add default argument
	 * TODO handle with try-catch-finally
	 * @param {Array} inputShape 
	 * @param {Object} optimizer 
	 * @param {string} loss 
	 * @param {Array} metrics 
	 */
	buildModel(inputShape, optimizer, loss, metrics) {

		// Define input, which has a size of inputShape
		const inputLayer = tf.input({ shape: inputShape });

		// Output dense layer uses linear activation.
		const denseLayer1 = tf.layers.dense({ 
			units: 1,
			kernelInitializer: 'zeros', 
			biasInitializer: 'zeros'
		});

		// Obtain the output symbolic tensor by applying the layers on the inputLayer.
		const output = denseLayer1.apply(inputLayer);

		// Create the model based on the inputs.
		this.model = tf.model({ inputs: inputLayer, outputs: output });

		this.model.compile({
			optimizer: optimizer,
			loss: loss,
			metrics: metrics,
		});
	}

}


class ARIMA {
	/**
	 * 
	 * @param {number} p 
	 * @param {number} d 
	 * @param {number} q 
	 */
	constructor(p, d, q) {
		this._p = p;
		this._d = d;
		this._q = q;
		this.arModel = new LinearRegression(this._p, "Auto Regression");
		this.maModel = new LinearRegression(this._q, "Moving Average");
	}

	get p() {
		return this._p;
	}

	get q() {
		return this._q;
	}

	get d() {
		return this._d;
	}

	/**
	 * 
	 * @param {tfjs.Tensor|Array} X 
	 * @param {object} params 
	 * 
	 * @returns {Promise<tf.History>}
	 */
	fit(X, params = {}) {
		this._keptFeatures = X;
		// Fitting AutoRegression(AR) Part
		return this.arModel.fit(X, params).then(() => {
			let [features, labels] = this.arModel.shiftInput(X);

			// Getting residuals (Noise) from AR
			let arPreds = this.arModel.predictSync(features, true);
			let residuals = tf.sub(labels, arPreds).arraySync();

			// Fitting MovingAverage (MA) Part
			return this.maModel.fit(residuals, params);
		});
	}


	/**
	 * 
	 * @param {Array|tfjs.Tensor|number} X 
	 * 
	 * @returns {Array}
	 */
	predictSync(X) {
		const arPreds = this.arModel.predictSync(X);
		const maPreds = this.maModel.predictSync(X);

		return tf.add(arPreds, maPreds).arraySync();
	}

	/**
	 * 
	 * @param {Array|tfjs.Tensor|number} X 
	 * 
	 * @returns {Promise<Array>}
	 */
	predict(X) {
		return Promise.resolve(this.predictSync(X));
	}

	/**
	 * 
	 * @param {tfjs.Tensor} yTrue 
	 * @param {tfjs.Tesnor} yPred 
	 * @param {function} fn 
	 * @returns 
	 */
	evaluate(yTrue, yPred, fn = tf.metrics.meanSquaredError) {
		return fn(yTrue, yPred);
	}

}


const opt = {
  "mqttUrl": "beta.masterofthings.com",
  "port": "1883",
  "topic": "iti/Mostafa",
  "username": "iti2021_projects",
  "password": "iti2021_projects",
  "message": ""
};

function MqttPublishAsync(msg){ 
  return new Promise(function(ok,cancel){
      opt["message"]=msg;
      MqttPublish(opt,function(err,result){
        if(err){
          return cancel(err);
          }
        ok();
       
     }); 
    });
}


// ARIMA TF Test
const X = tf.randomNormal([20], 50, 1).arraySync();
const trainSize = parseInt(0.8 * X.length);
const xTrain = X.slice(0, trainSize);
const xTest = X.slice(-1 * (X.length - trainSize));
// console.log(xTrain, xTest);

let params = {
    epochs: 10,
    batchSize: parseInt(0.2 * xTrain.length),
    validationSplit: .2,
    callbacks: tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 2 }),
    shuffle: false,
    verbose: 1
};

let arima = new ARIMA(2, 0, 0);

arima.fit(X, params).then(() => {
    return arima.predict(xTest.length);
})
.then(event.log)
.then(event.end())
.catch(err => {
    console.error(err);
    // event.end();
});





// Promise.all([
//   // MqttPublishAsync(JSON.stringify([1, 3])),
//   // MqttPublishAsync(JSON.stringify([1,2344])),
//   MqttPublishAsync("ARIMA Test")
// ])
// .then(function(){return event.end();})
// .catch(function(err){event.error(err)});
