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
			tf.train.adam(0.01),
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
		// TODO actually calculate periods
		return series.length;
	}

	/**
	 * 
	 * @param {Array|tfjs.Tensor} X 
	 * @param {Boolean} usingFeatures 
	 * 
	 * @returns {Array}
	 */
	predictSync(X, usingFeatures = false) {
		if (usingFeatures) {
			return this.model.predict(X).reshape([-1, 1]).arraySync();
		}
		let res = [...this._keptFeatures];
		let featureShape = [1, this._shifts];
		const sliceWindow = -1 * featureShape[1];
		if (this._shifts <= 0) {
			featureShape = [-1];
		}
		// TODO X should be a tensor of timestamp, then converted to periods
		let steps = LinearRegression.calculatePeriods(X); // FIXME steps should be calculated
		for (let s = 0; s < steps; s++) {
			let features = tf.tensor(res.slice(sliceWindow)).reshape(featureShape);
			let yHat = this.model.predict(features).arraySync();
			res.push(yHat[0][0]);
		}
		return res.slice(-steps);
	}

	/**
	 * 
	 * @param {Array|tfjs.Tensor} X 
	 * @param {Boolean} usingFeatures 
	 * 
	 * @returns {Promise<Array>}
	 */
	predict(X, usingFeatures = false) {
		return Promise.resolve(this.predictSync(X, usingFeatures));
	}

	/**
	 * 
	 * @param {number} this._shifts 
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

		let labels = X.slice(this._shifts);
        let features = [];
        for (let i = 1; i <= this._shifts; i++) {
			features.push(X.slice(this._shifts - i, -i));
        }

		return [
			tf.tensor(features).transpose(),
			tf.tensor(labels).reshape([-1, 1])
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
	buildModel(inputShape,
		optimizer = tf.train.adam(0.01),
		loss = "meanSquaredError",
		metrics = [tf.metrics.meanSquaredError]) {

		// Define input, which has a size of inputShape
		const inputLayer = tf.input({ shape: inputShape });

		// Output dense layer uses linear activation.
		const denseLayer1 = tf.layers.dense({ units: 1 });

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
	 * @param {Array|tfjs.Tensor} X 
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
	 * @param {Array|tfjs.Tensor} X 
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

module.exports = {
	ARIMA: (p, d, q) => { return new ARIMA(p, d, q); },
	ARMA: (p, q) => { return new ARIMA(p, q); },
	AR: (p) => { return new ARIMA(p, 0, 0); },
	MA: (q) => { return new ARIMA(0, 0, q); }
}