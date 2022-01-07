#! /usr/bin/env node
/**
 * 
 */
"use strict";

const np = require("./numpy");
const { GradientDescent } = require("./linreg");

// FIXME the ARIMA should extend linear regression instead
class AutoRegressionIntegratedMovingAverage extends GradientDescent {
	/**
	 * 
	 * @param {number} p 
	 * @param {number} d 
	 * @param {number} q 
	 * @param {number} learningRate 
	 * @param {object} KWArgs 
	 */
	constructor(p, d, q, learningRate = .001, KWArgs = {}) {
		super(learningRate, KWArgs);
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

	get theta() {
		return this._W.slice(0, this._p + 1);
	}

	set theta(value) {
		this._W = np.array([...value, ...this.phi]);
	}

	get phi() {
		return this._W.slice(-this._q);
	}

	set phi(value) {
		this._W = np.array([...this.theta, ...value]);
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
				np.arange(1, n + 1),//.reverse(),
				np.arange(n, X.length),
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
	 * FIXME only works with AR variants [AR, ARI, ARIMA]
	 * @param {Array|NDArray} X 
	 * @param {number} maxIter 
	 * @param {number} stopThreshold 
	 */
	fitSync(X, maxIter = 1024, stopThreshold = 1e-6) {
		this._initialValue = X.slice(-this._d - this._p, -this._p);
		var series = np.diff(X, this._d);
		var { labels, lags, residuals } = this._fitInit(series);
		var costOld = 0;
		var n = residuals.length;
		n = n ? n : lags.length;
		const ones = np.ones([lags.length, 1]);
		for (let epoch = 0; epoch < maxIter; epoch++) {
			var features = np.hstack([
				ones.slice(0, n),
				lags.slice(0, n),
				residuals
			]);
			var { costCurrent, gradient } = super._runEpoch(features, labels.slice(0, n));
			features = np.hstack([ones, lags]);
			var arW = this._W.slice(0, this._p + 1);
			residuals = labels.sub(np.dot(features, arW));
			residuals = this._buildPredictors(residuals, this._q);
			if (super._converged(costOld, costCurrent, stopThreshold, gradient)) {
				break;
			} else {
				costOld = costCurrent;
			}
		}
		if (this._q) {
			this._residuals = residuals[0].slice(-this._q);
		}
	}

	mleSync(X, maxIter = 1024, stopThreshold = 1e-6) {
		// calculate how many lags to keep to be able to forecast
		const keepLength = this._p + Math.max(this._q - 1, 0);
		// storing values for integration step during forecasting
		this._initialValue = X.slice(-this._d - keepLength, -keepLength);
		// differentiate, which gives labels
		var labels = np.diff(X, this._d);
		// init sequence
		let qLags;
		let Y;
		var pLags;
		let n;
		var costOld;
		({ qLags, Y, pLags, n, costOld, labels } = this._mleInit(labels, keepLength));
		// run for epochs number
		for (let epoch = 0; epoch < maxIter; epoch++) {
			var residuals = this._calculateResidulas(qLags, Y);
			// integrate residuals & lags in one feature matrix
			var features = np.hstack([pLags, residuals]);
			// run a single epoch
			var costCurrent;
			var gradient;
			({ costCurrent, gradient } = this._runEpochMLE(labels, features, qLags, pLags, n, Y));
			// check if convergence is achieved; update cost accordingly
			if (super._converged(costOld, costCurrent, stopThreshold, gradient)) {
				console.log(epoch);
				break;
			} else {
				costOld = costCurrent;
			}
		}
	}

	_calculateResidulas(qLags, Y) {
		// predict using only AR
		var X2 = np.dot(qLags, this.theta).T;
		// calculate residuals
		X2 = Y.sub(X2);
		return X2;
	}

	_runEpochMLE(labels, features, qLags, pLags, n, Y) {
		// declare variables for batch looping
		var end, batchX, batchY, batchPredictions, batchGradTheta, batchGradPhi, X;
		for (let start = 0; start < labels.length; start += this._b) {
			// set indices
			end = start + this._b;
			// slice into batches
			batchX = features.slice(start, end);
			batchY = labels.slice(start, end);
			// evaluate with current weights
			// batchPredictions = this.evaluate(batchX);
			batchPredictions = super.evaluate(batchX);
			// calculate error in prediction of ARMA
			var error = batchY.sub(batchPredictions);
			// calculate gradient of theta
			X = np.dot(qLags.T, this.phi).T;
			X = pLags.sub(X).T;
			batchGradTheta = np.dot(X, error);
			batchGradTheta = batchGradTheta.mul(this._alpha).div((this._b > 1) ? this._b : n); //.add(this._gamma*vt1);
			// update theta
			this.theta = this.theta.sub(batchGradTheta);
			// calculate gradient of phi
			X = this._calculateResidulas(qLags, Y).T;
			batchGradPhi = np.dot(X, error);
			batchGradPhi = batchGradPhi.mul(this._alpha).div((this._b > 1) ? this._b : n); //.add(this._gamma*vt1);

			// update phi
			this.phi = this.phi.sub(batchGradPhi);
		}
		var gradient = np.array([...batchGradTheta, ...batchGradPhi]);
		var costCurrent = this._costFn(batchY, batchPredictions, this._b);
		return { costCurrent, gradient };
	}

	_mleInit(labels, keepLength) {
		// build p lags
		var pLags = this._buildNLags(labels, this._p).T;
		// create a column of ones to calculate free/intercept term
		pLags = np.hstack([np.ones([pLags.length, 1]), pLags]);
		// build q lags of p lags for calculating residuals
		const qLags = this._buildNLags(pLags, this._q);
		// adjust dimensions
		const n = qLags.shape[1];
		// build q lags of labels for calculating residuals
		const Y = this._buildNLags(labels, this._q).T.slice(-n);
		labels = labels.slice(-n);
		pLags = pLags.slice(-n);
		// save lags to be used in forecasting
		this._lags = labels.slice(-keepLength);
		// init weights & batch size
		// this._W = (this._W) ? this._W : np.random.random([this._p + this._q + 1]);
		this._W = (this._W) ? this._W : np.zeros([this._p + this._q + 1]);
		this._b = this._b | labels.length;
		// init old cost for GD algorithm early stop
		var costOld = 0;
		return { qLags, Y, pLags, n, costOld, labels };
	}

	/**
	 * 
	 * @param {Array|NDArray} X 
	 * @returns 
	 */
	_fitInit(X) {
		var lags = this._buildPredictors(X, this._p);
		var labels = X.slice(this._p);
		this._W = np.zeros([this._p + this._q + 1, 1]);
		this._b = (this._b) ? this._b : labels.length;
		var residuals = this._buildPredictors(labels, this._q);
		this._lags = labels.slice(-this._p);
		labels = np.reshape(labels, [-1, 1]);
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
	 * 
	 * @param {number} periods 
	 * @returns 
	 */
	forecastSync(periods) {
		var lags = this._lags.slice();
		var residuals = [];
		if (this._residuals) {
			residuals = this._residuals.slice();
		}
		for (let i = 0; i < periods; i++) {
			var X = lags.slice(-this._p);
			X.push(...residuals.slice(-this._q));
			X.unshift(1);
			X = np.reshape(X, [1, -1]);
			var y = super.evaluate(X).flatten();
			lags.push(...y);
			if (residuals.length) {
				residuals.push(np.mean(residuals));
			}
			// TODO q elements
		}
		// the Integration step
		// https://stackoverflow.com/questions/43563241/numpy-diff-inverted-operation
		for (let d = this._d - 1; d >= 0; d--) {
			lags.unshift(this._initialValue[d]);
			lags = np.cumsum(lags);
		}
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
		(p, d, q, learningRate = .001, KWArgs = {}) => {
			return new AutoRegressionIntegratedMovingAverage(p, d, q, learningRate, KWArgs)
		},
}
