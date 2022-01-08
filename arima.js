#! /usr/bin/env node
/**
 * 
 */
"use strict";

const np = require("./numpy");
const { GradientDescent } = require("./linreg");

// FIXME the ARIMA should extend linear regression instead
class AutoRegressionIntegratedMovingAverage extends GradientDescent {

	constructor(order = [1, 0, 0], KWArgs = { learningRate: 1e-3, batchSize: 1 }) {
		// constructor(p, d, q, learningRate = 1e-3, KWArgs = { batchSize: 1 }) {
		// super(learningRate, KWArgs);
		super((KWArgs.learningRate) ? KWArgs.learningRate : 1e-3, KWArgs);
		[this._p, this._d, this._q] = order;
		// this._p = order[0];
		// this._d = order[1];
		// this._q = order[2];
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

	get theta() {
		return this._W.slice(-this._q);
	}

	set theta(value) {
		this._W = np.array([...this.theta, ...value]);
	}

	get phi() {
		return this._W.slice(0, this._p + 1);
	}

	set phi(value) {
		this._W = np.array([...value, ...this.phi]);
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
	 * TODO apply MLE on sigma for residuals
	 * @param {Array|NDArray} X 
	 * @param {number} maxIter 
	 * @param {number} stopThreshold 
	 */
	fitSync(X, maxIter = 1024, stopThreshold = 1e-6) {
		this._initialValue = X.slice(-this._d - this._p, -this._p);
		const yDiff = np.diff(X, this._d);
		var { labels, feats, residuals } = this._fitInit(yDiff);
		var costOld = 0;
		for (let epoch = 0; epoch < maxIter; epoch++) {
			var { costCurrent, gradient } = this._runEpoch(labels, feats, residuals);
			if (super._converged(costOld, costCurrent, stopThreshold, gradient)) {
				break;
			} else {
				costOld = costCurrent;
			}
		}
		if (this._q) {
			this._residuals = residuals.slice(-this._q);
		}
	}

	/**
	 * 
	 * @param {np.NDArray} labels 
	 * @param {np.NDArray} feats 
	 * @param {np.NDArray} residuals 
	 * @returns 
	 */
	_runEpoch(labels, feats, residuals) {
		var end, batchX, batchY, batchPredictions, gradient, features, error;
		for (let start = 0; start < labels.length; start += this._b) {
			features = np.hstack([
				feats,
				this._buildNLags(residuals, this._q),
			]);
			end = start + this._b;
			batchX = features.slice(start, end);
			batchY = labels.slice(start, end);
			batchPredictions = super.evaluate(batchX);
			error = batchY.sub(batchPredictions);
			residuals.splice(start + 1, error.length, ...error);
			gradient = batchX.T.dot(error);
			// TODO add nesterov update
			this._update(gradient, (this._b > 1) ? this._b : labels.length);
		}
		var costCurrent = this._costFn(batchY, batchPredictions, this._b);
		return { costCurrent, gradient };
	}

	/**
	 * 
	 * @param {np.NDArray} X 
	 * @returns 
	 */
	_fitInit(X) {
		this._W = np.random.random([this._p + this._q + 1]);
		const lags = this._buildNLags(X, this._p);
		const labels = X.slice(this._p);
		this._b = this._b | labels.length;
		var residuals = np.zeros([X.length]);
		this._lags = labels.slice(-this._p);
		const newAxis = np.ones([lags.length, 1]);
		const feats = np.hstack([newAxis, lags]);
		return { labels, feats, residuals };
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
		([p, d, q], KWArgs = { learningRate: 1e-3, batchSize: 1 }) => {
			// (p, d, q, learningRate = 1e-3, KWArgs = { batchSize: 1 }) => {
			// return new AutoRegressionIntegratedMovingAverage(p, d, q, learningRate, KWArgs)
			return new AutoRegressionIntegratedMovingAverage([p, d, q], KWArgs)
		},
}
