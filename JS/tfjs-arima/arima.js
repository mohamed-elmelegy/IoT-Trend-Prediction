#! /usr/bin/env node
/**
 *
 */
const tf = require("@tensorflow/tfjs");

class LinearRegression {
	DEFAULT_PARAMS = {
		epochs: 1024,
		shuffle: false,
		validationSplit: .2
	};

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
	 * Start a Linear model training
	 * @param {tfjs.Tensor|Array} X 
	 * @param {object} params 
	 * 
	 * @returns {Promise<tf.History>}
	 */
	fit(X, params = {}, learningRate=0.001) {
		let [features, labels] = this.shiftInput(X);
		const sliceWindow = -1 * this._shifts;
		this._keptFeatures = X.slice(sliceWindow);

		this.buildModel([features.shape[1]],
			tf.train.adam(learningRate),
			"meanSquaredError",
			[tf.metrics.meanSquaredError]
		);

		return this.model.fit(features, labels, params);
	}

	/**
	 * Predict values of a trained Linear model 
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
	 * Async Predict values of a trained Linear model 
	 * @param {Array|tfjs.Tensor|number} X 
	 * @param {Boolean} usingFeatures 
	 * 
	 * @returns {Promise<Array>}
	 */
	predict(X, usingFeatures = false) {
		return Promise.resolve(this.predictSync(X, usingFeatures));
	}

	/**
	 * Lag(shift) input data with order = this._shifts
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
	 * Build single unit neural network that acts like Linear model
	 * @param {Array} inputShape 
	 * @param {Object} optimizer 
	 * @param {string} loss 
	 * @param {Array} metrics 
	 */
	buildModel(inputShape,
		optimizer = tf.train.adam(0.01),
		loss = "meanSquaredError",
		metrics = [tf.metrics.meanSquaredError]) {

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
	 * Differentiate data with order = d (stationarization step)
	 * @param {tfjs.Tensor|Array} X 
	 * 
	 * @returns {Array}
	 */
	_diff(X) {
		if (this._d <= 0) {
			return [...X];
		}

		let diffX = [];
		for (let d = this._d; d < X.length; d++) {
			let value = X[d] - X[d - this._d];
			diffX.push(value);
		}
		return diffX;
	}

	/**
	 * Invert differentiated data with order = d, Destationarization Step used for predicted values
	 * @param {tfjs.Tensor|Array} X 
	 * 
	 * @returns {Array}
	 */
	_inverseDiff(X) {
		let history = [...this._keptFeatures];
		let res = [];
		for (let i = 0; i < X.length; i++) {
			let yHat = X[i];
			let historyIdx = history.length - this._d;
			let historyVal = history[historyIdx];
			let inverseVal = yHat + historyVal;

			res.push(inverseVal);
			history.push(inverseVal);
		}

		return res;
	}

	/**
	 * Start ARIMA model training
	 * @param {tfjs.Tensor|Array} X 
	 * @param {object} params 
	 * 
	 * @returns {Promise<tf.History>}
	 */
	fit(X, params = {}) {
		this._keptFeatures = X;
		let diffX = this._diff(X);
		// Fitting AutoRegression(AR) Part
		return this.arModel.fit(diffX, params).then(() => {
			let [features, labels] = this.arModel.shiftInput(diffX);

			// Getting residuals (Noise) from AR
			let arPreds = this.arModel.predictSync(features, true);
			let residuals = tf.sub(labels, arPreds).arraySync();

			// Fitting MovingAverage (MA) Part
			return this.maModel.fit(residuals, params);
		});
	}


	/**
	 * Predicts the next values by the number of steps given (X)
	 * @param {number} X 
	 * 
	 * @returns {Array}
	 */
	predictSync(X) {
		const arPreds = this.arModel.predictSync(X);
		const maPreds = this.maModel.predictSync(X);
		const arimaPreds = tf.add(arPreds, maPreds).arraySync();
		const finalPreds = this._inverseDiff(arimaPreds);
		return finalPreds;
	}

	/**
	 * Async Predicts the next values by the number of steps given (X)
	 * @param {number} X 
	 * 
	 * @returns {Promise<Array>}
	 */
	predict(X) {
		return Promise.resolve(this.predictSync(X));
	}

	/**
	 * Evaluate model performance using a given metric
	 * @param {tfjs.Tensor|Array} yTrue 
	 * @param {tfjs.Tesnor|Array} yPred 
	 * @param {function} fn 
	 * @returns 
	 */
	evaluate(yTrue, yPred, fn = tf.metrics.meanSquaredError) {
		return fn(yTrue, yPred);
	}

}

module.exports = {
	ARIMA: (p, d, q) => { return new ARIMA(p, d, q); },
	ARMA: (p, q) => { return new ARIMA(p, q); },
	AR: (p) => { return new ARIMA(p, 0, 0); },
	MA: (q) => { return new ARIMA(0, 0, q); }
}