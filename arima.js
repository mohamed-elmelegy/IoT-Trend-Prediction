#! /usr/bin/env node
/**
 * 
 */
"use strict";

const np = require("./numpy");
const { GradientDescent } = require("./linreg");

class AutoRegressionIntegratedMovingAverage extends GradientDescent {
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

	_buildPredictors(X, nCols) {
		var predictors = [];
		for (let idx = 1; idx <= nCols; idx++) {
			predictors.push(...X.slice(nCols - idx, -idx));
		}
		return np.reshape(predictors, [-1, nCols]);
	}

	score(X, y) {
		// TODO to be implemented
		throw new Error("Not Implemented yet!");
	}

	fitSync(X, maxIter = 1024, stopThreshold = 1e-6) {
		// FIXME working AR
		// var series = np.diff(X, this._d);
		// var n = Math.max(this._p, this._q);
		// var labels = series.slice(n);
		// var feats = this._buildPredictors(series, this._p, n);
		// this._lags = labels.slice(-this._p);
		// this._W = np.zeros([this._p + 1, 1]);
		// super.fitSync(feats, labels, maxIter, stopThreshold);
		// /FIXME
		this._initialValue = X[X.length - this._p - 1];
		var series = np.diff(X, this._d);
		var { lags, labels } = this._fitInit(series);
		var costOld = 0;
		var residuals = this._buildPredictors(labels, this._q);
		var n = residuals.length;
		n = n? n : lags.length;
		const ones = np.ones([lags.length, 1]);
		for (let epoch = 0; epoch < maxIter; epoch++) {
			var features = np.hstack([
				ones.slice(0, n),
				lags.slice(0, n),
				residuals
			]);
			// FIXME residuals should update with each epoch
			var [y, yHat, gradient] = super._runEpoch(features, labels.slice(0, n));
			// var costCurrent = super._costFn(y, yHat, this._b);
			var costCurrent = this._costFn(y, yHat, this._b);
			if (super._converged(costOld, costCurrent, stopThreshold, gradient)) {
				break;
			} else {
				costOld = costCurrent;
			}
			// residuals = (residuals.length) ? labels.sub(super.evaluate(features)) : residuals;
			// FIXME
			features = np.hstack([ones, lags]);
			var arW = this._W.slice(0, this._p + 1);
			residuals = labels.sub(np.dot(features, arW));
			residuals = this._buildPredictors(residuals, this._q);
		}
		// FIXME edge case
		if (this._q) {
			// this._lags.push(...residuals.slice(-this._q));
			this._residuals = residuals[0].slice(-this._q);//.flatten();
		}
	}

	_fitInit(X) {
		// var n = Math.max(this._p, this._q);
		// var n = this._p;
		var labels = X.slice(this._p);
		this._lags = labels.slice(-this._p);
		labels = np.reshape(labels, [-1, 1]);
		// var lags = this._buildPredictors(X, this._p, n);
		var lags = this._buildPredictors(X, this._p);
		this._W = np.zeros([this._p + this._q + 1, 1]);
		this._b = (this._b) ? this._b : labels.length;
		return { lags, labels };
	}

	_getResiduals(lags, labels) {
		var preds = super.evaluate(lags);
		var error = labels.sub(preds);
		var residuals = [];
		for (let idx = 1; idx <= this._q; idx++) {
			residuals.push(error.slice(this._q - idx, -idx));
		}
		residuals = np.transpose(residuals);
		return residuals;
	}

	async fit(X, maxIter = 1024, stopThreshold = 1e-6) {
		this.fitSync(X, maxIter, stopThreshold);
		return this;
	}

	predictSync(periods) {
		var lags = this._lags.slice();
		// var n = lags.length;
		var residuals = [];
		if (this._residuals) {
			residuals = this._residuals.slice();
			// n += residuals.length;
		}
		for (let i = 0; i < periods; i++) {
			var X = lags.slice(-this._p);
			X.push(...residuals.slice(-this._q));
			// var X = predictors.slice(-n);
			X.unshift(1);
			X = np.reshape(X, [1, -1]);
			var y = super.evaluate(X).flatten();
			lags.push(...y);
			if (residuals.length) {
				residuals.push(np.mean(residuals));
			}
			// TODO q elements
		}
		if (this._d) {
			lags.unshift(this._initialValue);
			lags = np.cumsum(lags);
		}
		return lags.slice(-periods);
	}

	async predict(periods) {
		return this.predictSync(periods);
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
