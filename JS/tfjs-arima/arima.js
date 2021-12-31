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
		const arSlice = -1 * (this._p + 1);
		this._keptFeatures = X.slice(arSlice);

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
	 * @returns 
	 */
	predictSync(X, usingFeatures = false) {
		if (usingFeatures) {
			return this.model.predict(X).reshape([-1, 1]).arraySync();
		}
		let res = this._keptFeatures;
		let arShape = [1, this._p + 1];
		const arSlice = -1 * arShape[1];
		if (this._p <= 0) {
			arShape = [-1];
		}
		// TODO X should be a tensor of timestamp, then converted to periods
		let steps = AutoRegression.calculatePeriods(X); // FIXME steps should be calculated
		for (let s = 0; s < steps; s++) {
			let features = tf.tensor(res.slice(arSlice)).reshape(arShape);
			let yHat = this.model.predict(features).arraySync();
			res.push(yHat[0][0]);
		}
		return res.slice(-steps);
	}

	/**
	 * 
	 * @param {Array|tfjs.Tensor} X 
	 * @param {Boolean} usingFeatures 
	 * @returns 
	 */
	predict(X, usingFeatures = false) {
		return Promise.resolve(this.predictSync(X, usingFeatures));
	}

	/**
	 * 
	 * @param {number} p 
	 * @param {Array} X 
	 * @returns 
	 */
	static pShift(p, X) {
		if (p <= 0) {
			return [
				tf.tensor(X).reshape([-1, 1]),
				tf.tensor(X).reshape([-1, 1])
			];
		}
		const featureShape = [X.length, p + 1];
		const labelShape = [-1, 1];
		let shift = 0;
		let shiftedXs = [];
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
			tf.tensor(shiftedXs).reshape(featureShape),
			tf.tensor([...X]).reshape(labelShape)
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
	DEFAULT_PARAMS = {
		epochs: 1024,
		shuffle: false,
		validationSplit: .2
	};

	constructor(q) {
		this._q = q;
	}

	get q() {
		return this._q;
	}

	fit(X, params = {}) {
		const maSlice = -1 * (this._q + 1);
		this._keptFeatures = X.slice(maSlice);
		
		let [features, labels] = MovingAverage.qShift(this._q, X);
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
	 * @returns 
	 */
	predictSync(X, usingFeatures = false) {
		if (usingFeatures) {
			return this.model.predict(X).reshape([-1, 1]).arraySync();
		}
		let res = this._keptFeatures;
		let maShape = [1, this._q + 1];
		const maSlice = -1 * maShape[1];
		if (this._q <= 0) {
			maShape = [-1];
		}
		// TODO X should be a tensor of timestamp, then converted to periods
		let steps = MovingAverage.calculatePeriods(X); // FIXME steps should be calculated
		for (let s = 0; s < steps; s++) {
			let features = tf.tensor(res.slice(maSlice)).reshape(maShape);
			let yHat = this.model.predict(features).arraySync();
			res.push(yHat[0][0]);
		}
		return res.slice(-steps);
	}

	/**
	 * 
	 * @param {Array|tfjs.Tensor} X 
	 * @param {Boolean} usingFeatures 
	 * @returns 
	 */
	predict(X, usingFeatures = false) {
		return Promise.resolve(this.predictSync(X, usingFeatures));
	}

	/**
	 * 
	 * @param {number} q 
	 * @param {Array} X 
	 * @returns 
	 */
	static qShift(q, X) {
		if (q <= 0) {
			return [
				tf.tensor(X).reshape([-1, 1]),
				tf.tensor(X).reshape([-1, 1])
			];
		}
		const featureShape = [X.length, q + 1];
		const labelShape = [-1, 1];
		let shift = 0;
		let shiftedXs = [];
		let outputArray = [...X];

		// TODO[danfojs]: replace this with danfojs
		while (shift <= q) {
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
			tf.tensor(shiftedXs).reshape(featureShape),
			tf.tensor([...X]).reshape(labelShape)
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
	DEFAULT_ARGS = {
		// TODO fill default arguments
	};
	constructor(p, d, q) {
		this._p = p;
		this._d = d;
		this._q = q;
		this.arModel = new AutoRegression(this._p);
		this.maModel = new MovingAverage(this._q);
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

	fit(X, params = {}) {
		this._keptFeatures = X;
		// Fitting AutoRegression(AR) Part
		return this.arModel.fit(X, params).then(() => {
			let [features, labels] = AutoRegression.pShift(this._p, X);

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
	 * @returns 
	 */
	predictSync(X) {
		const arPreds = this.arModel.predictSync(X);
		const maPreds = this.maModel.predictSync(X);

		return tf.add(arPreds, maPreds).arraySync();
	}

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

class ARMA extends ARIMA {
	constructor(p, q) {
		super(p, 0, q);
	}
}

module.exports = {
	AR: (p) => { return new AutoRegression(p); },
	MA: (q) => { return new MovingAverage(q); },
	ARIMA: (p, d, q) => { return new ARIMA(p, d, q); },
	ARMA: (p, q) => { return new ARMA(p, q); }
}