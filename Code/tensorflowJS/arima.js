#! /usr/bin/env node
/**
 *
 */
const tf = require("@tensorflow/tfjs");

class LinearRegression {

	/**
	 * Linear Model private fields
	 */
	#shifts = 0;
	#modelName = "";


	/**
	 * @param {number} shifts 
	 * @param {string} modelName 
	 */
	constructor(shifts, modelName = null) {
		if (!(Number.isInteger(shifts) && shifts >= 0)) {
			throw "Error: `shifts` must be 0 or positive Integer";
		}
		this.#shifts = shifts;
		this.#modelName = modelName;
	}

	get shifts() {
		return this.#shifts;
	}

	get modelName() {
		return this.#modelName;
	}

	/**
	 * Start a Linear model training
	 * @param {Array<number>} X 
	 * @param {object} params 
	 * 
	 * @returns {Promise<tf.History>}
	 */
	fit(X, params, learningRate) {
		let [features, labels] = this.shiftInput(X);
		const sliceWindow = -1 * this.#shifts;
		this._keptFeatures = X.slice(sliceWindow);

		// Build & compile Linear model
		const inputShape = [features.shape[1]];
		const optimizer = tf.train.adam(learningRate);
		const lossFunction = "meanSquaredError";
		const metrics = [tf.metrics.meanSquaredError];
		this.#buildModel(inputShape, optimizer, lossFunction, metrics);

		// Start training of Linear model 
		return this.model.fit(features, labels, params);
	}

	/**
	 * Predict values of a trained Linear model 
	 * X could be stepsNumber by default or features matrix when (usingFeatures=true)
	 * @param {Array<number>|tfjs.Tensor|number} X 
	 * @param {Boolean} usingFeatures 
	 * 
	 * @returns {Array<number>}
	 */
	predictSync(X, usingFeatures = false) {
		if (usingFeatures) {
			return this.model.predict(X).reshape([-1, 1]).arraySync();
		}
		let res = [...this._keptFeatures];
		let featureShape = [1, this.#shifts];
		const sliceWindow = -1 * featureShape[1];
		if (this.#shifts <= 0) {
			featureShape = [-1];
		}

		for (let s = 0; s < X; s++) {
			let features = tf.tensor(res.slice(sliceWindow)).reshape(featureShape);
			let yHat = this.model.predict(features).arraySync();
			res.push(yHat[0][0]);
		}
		return res.slice(-X);
	}

	/**
	 * Async Predict values of a trained Linear model 
	 * X could be stepsNumber by default or features matrix when (usingFeatures=true)
	 * @param {Array<number>|tfjs.Tensor|number} X 
	 * @param {Boolean} usingFeatures 
	 * 
	 * @returns {Promise<Array<number>>}
	 */
	predict(X, usingFeatures = false) {
		return Promise.resolve(this.predictSync(X, usingFeatures));
	}

	/**
	 * Lag(shift) input data with order = this._shifts
	 * @param {Array<number>} X 
	 * 
	 * @returns {Array<tf.Tensor>}
	 */
	shiftInput(X) {
		if (this.#shifts == 0) {
			return [
				tf.tensor(X).reshape([-1, 1]),
				tf.tensor(X).reshape([-1, 1])
			];
		}
		const featureShape = [(X.length - this.#shifts), this.#shifts];
		const labelShape = [-1, 1];
		let labels = X.slice(this.#shifts);
		let features = [];
		for (let i = 1; i <= this.#shifts; i++) {
			features.push(X.slice(this.#shifts - i, -i));
		}

		return [
			tf.tensor(features).reshape(featureShape),
			tf.tensor(labels).reshape(labelShape)
		];
	}

	/**
	 * Build single unit neural network that acts like Linear model
	 * @param {Array<number>} inputShape 
	 * @param {Object} optimizer 
	 * @param {string} lossFunction 
	 * @param {Array<tf.metrics>|Array<Function>} metrics 
	 */
	#buildModel(inputShape,
		optimizer = tf.train.adam(1e-3),
		lossFunction = "meanSquaredError",
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

		// Create the model based on the input shape.
		this.model = tf.model({ inputs: inputLayer, outputs: output });

		this.model.compile({
			optimizer: optimizer,
			loss: lossFunction,
			metrics: metrics,
		});
	}

}


class ARIMA {
	
	/**
	 * ARIMA private fields
	 */
	#p = 0;
	#d = 0;
	#q = 0;
	#arModel = {};
	#maModel = {};

	#DEFAULT_PARAMS = {
		epochs: 2048,
		shuffle: false, // time series is ordered
		validationSplit: .2, // cross validation 20% test -> validation score 
		callbacks: tf.callbacks.earlyStopping({
			monitor: 'val_loss',
			patience: 3
		})
	};

	#LEARNING_RATE = 1e-2;

	/**
	 * 
	 * @param {number} p 
	 * @param {number} d 
	 * @param {number} q 
	 */
	constructor(p, d, q) {
		this.#p = p;
		this.#d = d;
		this.#q = q;
		this.#arModel = new LinearRegression(this.#p, "Auto Regression");
		this.#maModel = new LinearRegression(this.#q, "Moving Average");
	}

	get p() {
		return this.#p;
	}

	get q() {
		return this.#q;
	}

	get d() {
		return this.#d;
	}

	get arModel() {
		return this.#arModel;
	}

	get maModel() {
		return this.#maModel;
	}

	/**
	 * Differentiate data with order = d (stationarization step)
	 * @param {Array<number>} X 
	 * 
	 * @returns {Array<number>}
	 */
	#diff(X) {
		if (this.#d <= 0) {
			return [...X];
		}

		let diffX = [];
		for (let d = this.#d; d < X.length; d++) {
			let value = X[d] - X[d - this.#d];
			diffX.push(value);
		}
		return diffX;
	}

	/**
	 * Invert differentiated data with order = d, De-Stationarization Step used for predicted values
	 * @param {Array<number>} X 
	 * 
	 * @returns {Array<number>}
	 */
	#inverseDiff(X) {
		if (this.#d <= 0) {
			return X;
		}
		let history = [...this._keptFeatures];
		let res = [];
		for (let i = 0; i < X.length; i++) {
			let yHat = X[i];
			let historyIdx = history.length - this.#d;
			let historyVal = history[historyIdx];
			let inverseVal = yHat + historyVal;

			res.push(inverseVal);
			history.push(inverseVal);
		}

		return res;
	}

	/**
	 * Start ARIMA model training
	 * @param {Array<number>} X 
	 * @param {object} params 
	 * @param {number} learningRate 
	 * 
	 * @returns {Promise<tf.History>}
	 */
	fit(X, params = this.#DEFAULT_PARAMS, learningRate=this.#LEARNING_RATE) { 
		// inputs => neuron(AR) => neuron(MA) => output
		// pureJS: inputs (extract residuals) => model(AR,MA) => output
		// tfJS: inputs => model(AR, MA:const) model(AR:const, MA) => output
		// y_t = phi_p y_t-p + theta_q e_t-q;
		// e_t = y_t - y'_t (phi_p y_t-p + theta_q e_t-q=0)
		this._keptFeatures = X;
		let yPrime = this.#diff(X);
		// Fitting AutoRegression(AR) Part
		return this.#arModel.fit(yPrime, params, learningRate).then(() => {
			let [features, labels] = this.#arModel.shiftInput(yPrime);

			// Getting residuals (Noise) from AR
			let arPreds = this.#arModel.predictSync(features, true); // predictSync of LinearRegression
			let residuals = tf.sub(labels, arPreds).arraySync(); // observed

			// Fitting MovingAverage (MA) Part
			return this.#maModel.fit(residuals, params, learningRate);
		});
	}


	/**
	 * Predicts the next values by the number of steps given (stepsNumber)
	 * @param {number} stepsNumber 
	 * 
	 * @returns {Array<number>}
	 */
	predictSync(stepsNumber=1) {
		// Predict AutoRegressive results (values)
		const arPreds = this.#arModel.predictSync(stepsNumber);

		// Predict MovingAverage results (residuals)
		const maPreds = this.#maModel.predictSync(stepsNumber);

		// Get AR output + MA output 
		const arimaPreds = tf.add(arPreds, maPreds).arraySync(); // Array(10) arr1[i]+arr2[i] 

		// Inverse diff operation which happened before fitting the model
		// to get back prediction values in the same range of input data
		const finalPreds = this.#inverseDiff(arimaPreds);

		return finalPreds;
	}

	/**
	 * Async Predicts the next values by the number of steps given (stepsNumber)
	 * @param {number} stepsNumber 
	 * 
	 * @returns {Promise<Array>}
	 */
	predict(stepsNumber=1) {
		return Promise.resolve(this.predictSync(stepsNumber));
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