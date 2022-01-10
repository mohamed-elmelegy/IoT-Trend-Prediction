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

	constructor(order = [1, 0, 0], KWArgs = { learningRate: 1e-3 }) {
		super(KWArgs.learningRate || 1e-3, KWArgs);
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
		return this.mu * (1 - this.phi.sum());
	}

	get theta() {
		return this._W.slice(-this._q);
	}

	get phi() {
		return this._W.slice(0, this._p);
	}

	/**
	 * 
	 * @param {Array} X 
	 * @param {number} n 
	 * @returns 
	 */
	_buildNLags(X, n = 0) {
		return np.linalg.toeplitz(X)
			.at(
				np.arange(n, X.length),
				np.arange(1, n + 1),
			);
	}

	/**
	 * 
	 * @param {Array|NDArray} X 
	 * @param {number} nCols 
	 * @returns 
	 */
	_buildPredictors(X, nCols) {
		var predictors = [];
		for (let idx = 1; idx <= nCols; idx++) {
			predictors.push(...X.slice(nCols - idx, -idx));
		}
		return np.reshape(predictors, [-1, nCols]);
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
	 * FIXME only works with AR variants: ARIMA(p,d,0), ARIMA(p,d,q)
	 * @param {Array|NDArray} X 
	 * @param {number} maxIter 
	 * @param {number} stopThreshold 
	 */
	fitSync(X, maxIter = 1024, stopThreshold = 1e-6) {
		// keeping values for integration step
		this._initialValue = X.slice(-this._d - this._p, -this._p);
		// difference the data
		const yDiff = np.diff(X, this._d);
		// initialise the fit subroutine
		var { labels, lags, residuals } = this._fitInit(yDiff);
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
	 * @param {Array|NDArray} labels 
	 * @param {Array|NDArray} residuals 
	 */
	_calculateMetrics(labels, residuals) {
		// number of observation
		const n = labels.length;
		// number of model parameters
		const k = 1 + (this.intercept != 0) + this._p + this._q;
		// variance of the residuals white noise
		this.sigma2 = residuals.power(2).sum() / (n - this._p);
		// initialise metrics object
		this.metrics = {};
		// calculating log likelihood of the data
		this.metrics.LL = -(n / 2) * (1 + Math.log(2 * Math.PI * this.sigma2));
		// calculating Akaike's Information Criteria
		this.metrics.AIC = 2 * (k - this.metrics.LL);
		// calculating corrected AIC
		this.metrics.AICc = this.metrics.AIC + 2 * k * (k + 1) / (n - k - 1);
		// calculating Bayesian Information Criteria
		this.metrics.BIC = this.metrics.AIC + k * (Math.log(n) - 2);
	}

	/**
	 * 
	 * @param {np.NDArray} labels 
	 * @param {np.NDArray} feats 
	 * @param {np.NDArray} residuals 
	 * @returns 
	 */
	_runEpoch(labels, feats, residuals) {
		// define variables required
		var end, batchX, batchY, batchPredictions, gradient, features, error;
		// loop over the data in batches; if _b == labels.length -> Vanilla GD; _b == 1 -> stochastic GD
		for (let start = 0; start < labels.length; start += this._b) {
			// combine lags & residuals into a single matrix for vectorised operation
			features = np.hstack([
				feats,
				this._buildNLags(residuals, this._q),
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
			residuals.splice(start + 1, error.length, ...error);
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
	 * @param {np.NDArray} X 
	 * @returns 
	 */
	_fitInit(X) {
		/**
		 * random initialise for weights
		 * pros: could converge faster, worst case scenario is to take as much as 
		 * zero initialisation
		 * cons: easy to stuck at local minima
		 */
		this._W = np.random.random([this._p + this._q]);
		// TODO models could be without mean/constant
		this.mu = X.mean();
		X = X.sub(this.mu);
		// build lags AKA feature vector for AR
		const lags = this._buildNLags(X, this._p);
		// set labels
		const labels = X.slice(this._p);
		// determine the batch size
		this._b = this._b || labels.length;
		// initialise residuals
		// FIXME tested well against ARIMA(p, d, 0), ARIMA(p, d, q); might not against ARIMA(0, d, q)
		var residuals = np.zeros([X.length]);
		// keep lags for forecasting
		this._lags = labels.slice(-this._p);
		// return required data
		return { labels, lags, residuals };
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
	 * TODO forecast needs work; residuals to be drawn from gaussian
	 * @param {number} periods 
	 * @returns 
	 */
	forecastSync(periods) {
		// load required lags for forecasting
		var lags = this._lags.slice();
		// for ARIMA(p, d, 0), initialise residuals to empty array
		var residuals = [];
		// for ARIMA(p, d, q)
		// TODO ARIMA(0, d, q)
		if (this._residuals) {
			residuals = this._residuals.slice();
		}
		// predict periods one by one
		for (let i = 0; i < periods; i++) {
			// prepare a single record for forecasting
			var X = [...lags.slice(-this._p), ...residuals.slice(-this._q)];
			// forecasting
			var y = this.evaluate(X) + this.intercept;
			// adding white noise
			var [e] = tf.randomNormal([1], 0, Math.sqrt(this.sigma2)).arraySync();
			y += e;
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
		return lags.slice(-periods);
	}

	/**
	 * 
	 * @param {number} periods 
	 * @returns 
	 */
	async forecast(periods) {
		return this.forecastSync(periods);
	}

	updateSync(trueLags) {
		// TODO not implemented yet
		throw new Error("Not Implemented yet!");
	}

	async update(trueLags) {
		return this.updateSync(trueLags);
	}
}

module.exports = {
	ARIMA:
		([p, d, q], KWArgs = { learningRate: 1e-3 }) => {
			return new AutoRegressionIntegratedMovingAverage([p, d, q], KWArgs)
		},
}
