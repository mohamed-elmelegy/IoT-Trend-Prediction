#! /usr/bin/env node
/**
 *
 */
const tf = require("@tensorflow/tfjs");

class AutoRegression {
	DEFAULT_PARAMS = {
		epochs: 1024,
		shuffle: false,
		validationSplit: .2
	};
	/**
	 * 
	 * @param {number} p 
	 */
	constructor(p) {
		this._p = p;
	}
	get p() {
		return this._p;
	}

	/**
	 * 
	 * @param {tfjs.Tensor|Array} X 
	 * @param {object} params 
	 * @returns 
	 */
	fit(X, params = {}) {
		let [features, labels] = AutoRegression.pShift(this._p, X);

		// FIXME cross-validation might be required
		// TODO for cross validation: https://js.tensorflow.org/api/latest/#tf.LayersModel.fit
		this.buildModel([features.shape[1]],
			tf.train.adam(0.01),
			"meanSquaredError",
			[tf.metrics.meanSquaredError]
		);

		this._keptFeatures = X.slice(-this._p);

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
	 * @returns 
	 */
	predictSync(X) {
		let res = this._keptFeatures;
		const shape = [1, this._p];
		// TODO X should be a tensor of timestamp, then converted to periods
		let steps = AutoRegression.calculatePeriods(X); // FIXME steps should be calculated
		for (let s = 0; s < steps; s++) {
			let features = tf.tensor(res.slice(-this._p)).reshape(shape);
			let yHat = this.model.predict(features).arraySync();
			res.push(yHat[0]);
		}
		return res.slice(-steps);
	}

	/**
	 * 
	 * @param {Array|tfjs.Tensor} X 
	 * @returns 
	 */
	predict(X) {
		return Promise.resolve(this.predictSync(X));
	}

	/**
	 * 
	 * @param {number} p 
	 * @param {Array} X 
	 * @returns 
	 */
	static pShift(p, X) {
		if (p <= 0) {
			return [X, X]
		}
		let shift = 1;
		let shiftedXs = [];
		let inputArray = [...X];
		let outputArray = [...X];
		// TODO[danfojs]: replace this with danfojs
		while (shift <= p) {
			let shiftedX = [];
			for (let k = 0; k < shift; k++) {
				// Add temporary 0 instead of NAN... 
				// TODO: Input Data should be cleaned
				shiftedX.push(0);
			}
			for (let i = 0; i < outputArray.length - shift; i++) {
				shiftedX.push(X[i]);
			}
			outputArray = shiftedX;
			shiftedXs.push(shiftedX);
			shift += 1;
		}

		return [
			tf.tensor(shiftedXs).transpose(),
			tf.tensor(inputArray).reshape([-1, 1])
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

class MovingAverage {
	DEFAULT_ARGS = {
		// TODO fill default arguments
	};
	constructor(q) {
		this._q = q;
	}

	get q() {
		return this._q;
	}

	fitSync(X, y, args = {}) {
		// TODO fill the fit method
	}

	fit(X, y, args = {}) {
		return Promise.resolve(this.fitSync(X, y, args));
	}

	predictSync(toPredict) {
		// TODO fill me
		return this.model.predict(toPredict);
	}

	predict(toPredict) {
		return Promise.resolve(this.predict(toPredict));
	}

}

class ARIMA {
	DEFAULT_ARGS = {
		// TODO fill default arguments
	};
	constructor(p, d, q) {
		this._p = p;
		this._d = d;
		this._q = q;
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


	fitSync(X, y, args = {}) {
		// TODO fill the fit method
	}

	fit(X, y, args = {}) {
		return Promise.resolve(this.fitSync(X, y, args));
	}

	predictSync(toPredict) {
		// TODO handle toPredict
		return this.model.predict(toPredict);
	}

	predict(toPredict) {
		return Promise.resolve(this.predictSync(toPredict));
	}
}

class ARMA extends ARIMA {
	constructor(p, q) {
		super(p, 0, q);
	}
}

module.exports = {
	AR: (p) => { return new AutoRegression(p); },
	MA: (q) => { return new MovingAverage(q); },
	ARIMA,
	ARMA
}