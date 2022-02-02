#! /usr/bin/env node
/**
 * 
 */
"use strict";

const np = require("./numpy");
const tf = require("@tensorflow/tfjs");
const { GradientDescent } = require("./linreg");

// FIXME the ARIMA should extend linear regression instead
class AutoRegressionIntegratedMovingAverage extends GradientDescent {

	/**
	 * TODO ARIMA could be of order (0, 0, 0)
	 * @param {Array|NDArray} order 
	 * @param {object} KWArgs 
	 */
	constructor(order, KWArgs = { learningRate: 1e-1 }) {
		super(KWArgs.learningRate || 1e-1, KWArgs);
		[this._p, this._d, this._q] = order;
		this._update = function (gradient, m, vt1 = 0) {
			this._W = this._W.add(this.vt(gradient, m, vt1));
		};
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

	get intercept() {
		// FIXME this is different from equations, yet it matches python results
		return this.mu; // * (1 - this.phi.sum());
	}

	get theta() {
		if (this._q) {
			return this._W.slice(-this._q);
		}
	}

	get phi() {
		if (this._p) {
			return this._W.slice(0, this._p);
		}
	}

	/**
	 * 
	 * @param {Array} X 
	 * @param {number} n 
	 * @returns 
	 */
	static _buildNLags(X, n = 0) {
		return n ? np.linalg.toeplitz(X)
			.at(
				np.arange(n, X.length),
				np.arange(1, n + 1),
			) : [];
	}

	/**
	 * 
	 * @param {Array|NDArray} X 
	 * @param {Array|NDArray} y 
	 */
	score(X, y) {
		// TODO to be implemented
		throw new Error("Not Implemented yet!");
	}

	/**
	 * 
	 * @param {Array|NDArray} X 
	 * @param {number} maxIter 
	 */
	fitStatSync(X, maxIter = 1024) {
		// initialise the fit subroutine
		var { labels, lags, residuals } = this._fitInit(np.array(X));
		this._runEpoch(labels, lags, residuals);
		var weights = np.array([this._W]);
		var stop = 0;
		const size = [32];
		const replace = labels.length < 32;
		const batch = this._b;
		this._b = X.length;
		// loop for specified epoch number
		for (let epoch = 0; epoch < maxIter; epoch++) {
			var idx = np.random.choice(labels.length, size, replace);
			// run a single epoch, get the final cost & gradient
			var batchX, batchY, batchPredictions, gradient, features, error;
			// combine lags & residuals into a single matrix for vectorised operation
			features = np.hstack([
				lags,
				AutoRegressionIntegratedMovingAverage._buildNLags(residuals, this._q),
			]);
			// slicing data as batches
			batchX = features.at(idx)
			batchY = labels.at(idx)
			// predicting by batch
			batchPredictions = super.evaluate(batchX);
			// calculating error AKA residual
			error = batchY.sub(batchPredictions);
			// updating residuals vector
			// TODO this takes time, would be better w/ partial update
			residuals.splice(
				this._q,
				labels.length,
				...labels.sub(super.evaluate(features))
			);
			// calculating gradient
			gradient = batchX.T.dot(error);
			// TODO add nesterov update
			this._update(gradient, (this._b > 1) ? this._b : labels.length);
			// }
			// calculate cost after one epoch
			var costOld = weights.mean();
			// post-epoch
			weights.push(this._W);
			var costCurrent = weights.mean();
			// early stopping condition 0.00001
			// TODO the condition should be on each weight, not collective mean of them
			const converged = this._converged(costOld, costCurrent, 1e-5, gradient);
			stop += converged;
			if (stop >= 4) {
				break;
			} else {
				if (!converged) {
					stop = 0;
				}
				// update cost for next epoch
				costOld = costCurrent;
			}
		}
		this._b = batch;
		this._W = np.mean(weights, 0);
		this._calculateMetrics(labels, residuals);
		// keeping residuals for forecasting purposes
		if (this._q) {
			this._residuals = residuals.slice(-this._q);
		}
	}

	/**
	 * 
	 * @param {Array|NDArray} X 
	 * @param {number} maxIter 
	 * @returns 
	 */
	async fitStat(X, maxIter = 1024) {
		this.fitStatSync(X, maxIter);
		return this;
	}

	/**
	 * X must be preprocessed to have constant frequency
	 * @param {Array|NDArray} X 
	 * @param {number} maxIter 
	 * @param {number} stopThreshold 
	 */
	fitSync(X, maxIter = 1024, stopThreshold = 1e-6) {
		// initialise the fit subroutine
		var { labels, lags, residuals } = this._fitInit(np.array(X));
		// initialise cost
		var costOld = 0;
		// loop for specified epoch number
		for (let epoch = 0; epoch < maxIter; epoch++) {
			// run a single epoch, get the final cost & gradient
			var { costCurrent, gradient } = this._runEpoch(labels, lags, residuals);
			// early stopping condition
			if (super._converged(costOld, costCurrent, stopThreshold, gradient)) {
				break;
			} else {
				// update cost for next epoch
				costOld = costCurrent;
			}
		}
		this._calculateMetrics(labels, residuals);
		// keeping residuals for forecasting purposes
		if (this._q) {
			this._residuals = residuals.slice(-this._q);
		}
	}

	/**
	 * 
	 * @param {Array|NDArray} X 
	 * @returns 
	 */
	_fitInit(X) {
		/**
		 * random initialise for weights
		 * pros: could converge faster, worst case scenario is to take as much as 
		 * zero initialisation
		 * cons: easy to stuck at local minima
		 */
		this._W = this._W || np.random.random([this._p + this._q]);
		var yPrime = X.slice();
		this.mu = yPrime.mean();
		this._observationsNumber = yPrime.length;
		// preprocessing (mean) to make mean = 0
		yPrime = yPrime.sub(this.mu);
		// keeping values for integration step
		this._initialValue = [];
		// difference the data
		for (let d = 0; d < this._d; d++) {
			this._initialValue.push(yPrime.at(-1 - this._p));
			yPrime = np.diff(yPrime);
		}
		// build lags AKA feature vector for AR
		const lags
			= AutoRegressionIntegratedMovingAverage._buildNLags(yPrime, this._p);
		// set labels
		var labels = yPrime.slice(this._p || this._q);
		// determine the batch size
		this._b = this._b || labels.length;
		var residuals = [];
		// initialise residuals by noise; because we still initialising
		residuals
			= tf.randomNormal(labels.shape, labels.mean(), labels.std())
				.arraySync();
		if (this._p) {
			residuals.unshift(...np.zeros(this._q));
		} else {
			residuals
				.unshift(...tf
					.randomNormal([this._q], labels.mean(), labels.std())
					.arraySync());
		}
		residuals = np.array(residuals);
		// keep lags for forecasting
		this._lags = labels.slice(-this._p || labels.length);
		return { labels, lags, residuals };
	}

	/**
	 * 
	 * @param {np.NDArray} labels 
	 * @param {np.NDArray} lags 
	 * @param {np.NDArray} residuals 
	 * @returns 
	 */
	_runEpoch(labels, lags, residuals) {
		// define variables required
		var end, batchX, batchY, batchPredictions, gradient, features, error;
		// loop over the data in batches; if _b == labels.length -> Vanilla GD
		// _b == 1 -> stochastic GD
		for (let start = 0; start < labels.length; start += this._b) {
			// combine lags & residuals into a single matrix for vectorised operation
			features = np.hstack([
				lags,
				AutoRegressionIntegratedMovingAverage._buildNLags(residuals, this._q),
			]);
			// calculate indices
			end = start + this._b;
			// slicing data as batches
			batchX = features.slice(start, end);
			batchY = labels.slice(start, end);
			// predicting by batch
			batchPredictions = super.evaluate(batchX);
			// calculating error AKA residual
			error = batchY.sub(batchPredictions);
			// updating residuals vector
			residuals.splice(start + this._q, error.length, ...error);
			// calculating gradient
			gradient = batchX.T.dot(error);
			// TODO add nesterov update
			this._update(gradient, (this._b > 1) ? this._b : labels.length);
		}
		// calculate cost after one epoch
		var costCurrent = this._costFn(batchY, batchPredictions, this._b);
		// return required data
		return { costCurrent, gradient };
	}

	/**
	 * 
	 * @param {Array|NDArray} labels 
	 * @param {Array|NDArray} residuals 
	 */
	_calculateMetrics(labels, residuals) {
		// number of observation
		const T = labels.length;
		// number of model parameters
		// k = (this.intercept != 0) from slides
		const k = 1 + (this.intercept != 0) + this._p + this._q;
		// variance of the residuals white noise
		this.sigma2 = residuals.power(2).sum() / T;
		// initialise metrics object
		this.metrics = {};
		// calculating log likelihood of the data
		this.metrics.LL = -(T / 2) * (1 + Math.log(2 * Math.PI * this.sigma2));
		// calculating Akaike's Information Criteria
		this.metrics.AIC = 2 * (k - this.metrics.LL);
		// calculating corrected AIC
		this.metrics.AICc = this.metrics.AIC + 2 * k * (k + 1) / (T - k - 1);
		// calculating Bayesian Information Criteria
		this.metrics.BIC = this.metrics.AIC + k * (Math.log(T) - 2);
	}

	/**
	 * 
	 * @param {Array|NDArray} X 
	 * @param {number} maxIter 
	 * @param {number} stopThreshold 
	 * @returns 
	 */
	async fit(X, maxIter = 1024, stopThreshold = 1e-6) {
		this.fitSync(X, maxIter, stopThreshold);
		return this;
	}

	/**
	 * 
	 * @param {number} periods 
	 * @returns 
	 */
	forecastSync(periods) {
		// load required lags for forecasting
		var lags = this._lags.slice();
		// for ARIMA(p, d, 0), initialise residuals to empty array
		var residuals = [];
		// for ARIMA(p, d, q)
		if (this._residuals) {
			residuals = this._residuals.slice();
		}
		// predict periods one by one
		for (let i = 0; i < periods; i++) {
			// prepare a single record for forecasting
			var X = [
				...lags.slice(-this._p || lags.length),
				...residuals.slice(-this._q)
			];
			// forecasting
			var y = this.evaluate(X); //+ (this.intercept || 0);
			// adding white noise
			var [e] = tf.randomNormal([1], 0, Math.sqrt(this.sigma2)).arraySync();
			// y += e;
			// update lags for next forecast
			lags.push(y);
			// for ARIMA(p, d, q) update residuals as well
			if (residuals.length) {
				residuals.push(e);
			}
		}
		// the Integration step
		// https://stackoverflow.com/questions/43563241/numpy-diff-inverted-operation
		for (let d = this._d - 1; d >= 0; d--) {
			lags.unshift(this._initialValue[d]);
			lags = np.cumsum(lags);
		}
		// return the forecasted values
		// return lags.slice(-periods);
		return lags.add(this.intercept || 0).slice(-periods);
	}

	/**
	 * 
	 * @param {number} periods 
	 * @returns 
	 */
	async forecast(periods) {
		return this.forecastSync(periods);
	}

	/**
	 * 
	 * @param {Array|NDAarray} trueLags 
	 */
	updateSync(trueLags) {
		// update init sequence
		{
			var labels = np.array(trueLags);
			this.mu
				= ((this.mu * this._observationsNumber) + labels.sum())
				/ (this._observationsNumber + labels.length);
			this._observationsNumber += labels.length;
			labels = labels.sub(this.mu);
			this._b = (parseInt(this._b / labels.length)) ? labels.length : this._b;
			var lags = AutoRegressionIntegratedMovingAverage
				._buildNLags(np.array([...this._lags, ...labels]), this._p);
			var yHat = this.forecastSync(labels.length);
			var residuals = this._residuals || np.array([]);
			residuals.push(...labels.sub(yHat));
			// initialise cost
			var { costCurrent: costOld } = this._runEpoch(labels, lags, residuals);
		}
		// loop for specified epoch number
		// FIXME when researching code in python, this whole function is a workaround
		for (let epoch = 0; epoch < 32; epoch++) {
			// run a single epoch, get the final cost & gradient
			var { costCurrent, gradient } = this._runEpoch(labels, lags, residuals);
			// early stopping condition
			if (super._converged(costOld, costCurrent, 1e-4, gradient)) {
				break;
			} else {
				// update cost for next epoch
				costOld = costCurrent;
			}
		}
		var temp = this.metrics;
		this._calculateMetrics(labels, residuals);
		// updating metrics subroutine
		{
			this.metrics.LL += temp.LL;
			this.metrics.AIC += temp.AIC;
			this.metrics.AICc += temp.AICc;
			this.metrics.BIC += temp.BIC;
		}		// keeping residuals for forecasting purposes
		if (this._q) {
			this._residuals = residuals.slice(-this._q);
		}
	}

	/**
	 * 
	 * @param {Array|NDArray} trueLags 
	 * @returns 
	 */
	async update(trueLags) {
		this.updateSync(trueLags);
		return this;
	}
}

module.exports = {
	/**
	 * 
	 * @param {Array} order order of ARIMA in (p, d, q) format
	 * @param {object} KWArgs 
	 * @returns 
	 */
	ARIMA:
		([p, d, q], KWArgs = { learningRate: 1e-1, batchSize: 1 }) => {
			return new AutoRegressionIntegratedMovingAverage([p, d, q], KWArgs)
		},
	arima: AutoRegressionIntegratedMovingAverage
}
