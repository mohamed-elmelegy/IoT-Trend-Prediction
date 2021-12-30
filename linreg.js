#! /usr/bin/node
/**
 * TODO update the file to match tensorflow
 */
"use strict";

const np = require("./numpy");

class GradientDescent {
	/**
	 * 
	 * @param {number} learningRate 
	 * @param {object} kwargs 
	 */
	constructor(learningRate = 0.001, kwargs = {}) {
		this._alpha = learningRate;
		this._W = kwargs["weights"];
		this._gamma = kwargs["momentum"] | 0;
		this._b = kwargs["batchSize"];
		this._costFn = kwargs["costFunction"];
		if (!this._costFn) {
			this._costFn = function (labels, predictions, m = null) {
				m = 2 * ((m) ? m : labels.length);
				return np.sum(labels.sub(predictions).power(2)) / m;
			};
		}
		// FIXME gradient is d[cost]/d_W, so it should be different for each cost function
		this._grad = kwargs["gradient"];
		if (!this._grad) {
			this._grad = function (X, y, yHat) {
				var error = yHat.sub(y);
				return np.dot(np.transpose(X), error);
			}
		}
		// TODO nesterov update
		this._update = (kwargs["nesterov"]) ? this.updateNesterov : function (gradient, m, vt1 = 0) {
			this._W = this._W.sub(this.vt(gradient, m, vt1));
		};
	}

	set alpha(learningRate) {
		this._alpha = learningRate;
	}

	get alpha() {
		return this._alpha;
	}

	set gamma(momentum) {
		this._gamma = momentum;
	}

	get gamma() {
		return this._gamma;
	}

	get _coef() {
		return this._W;
	}

	/**
	 * 
	 * @param {NDArray} X 
	 * @returns 
	 */
	evaluate(X) {
		return np.dot(X, this._W);
	}

	updateNesterov(X, y, m, vt1) {
		// TODO implement nesterov's update
		throw Error("Method not implemented yet")
	}

	vt(gradient, m, vt1 = 0) {
		return gradient.mul(this._alpha)
			.div(m)
			.add(this._gamma * vt1);
	}

	/**
	 * 
	 * @param {NDArray} X 
	 * @returns 
	 */
	async predict(X) {
		var features = X.slice();
		if (np.ndim(features) == 1) {
			features = np.reshape(features, [-1, 1]);
		}
		features = [np.ones([np.shape(features)[1], 1]), features];
		features = np.hstack(features);
		return this.evaluate(features);
	}

	fitSync(X, y, maxIter = 1024, stopThreshold = 1e-6) {
		// let ut = 0 // TODO support adaptive grad
		var costOld;
		({ costOld, y, X } = this._fitInit(X, y));
		for (let epoch = 0; epoch < maxIter; epoch++) {
			var [yBatch, yHatBatch, gradientBatch] = this._runEpoch(X, y);
			var costCurrent = this._costFn(yBatch, yHatBatch, this._b);
			if (this._converged(costOld, costCurrent, stopThreshold, gradientBatch)) {
				break;
			} else {
				costOld = costCurrent;
			}
		}
	}

	_fitInit(X, y) {
		var nRows = y.length;
		this._b = (this._b) ? this._b : nRows;
		// FIXME
		// var costOld = this._costFn(y.slice(-this._b), np.zeros([this._b]), this._b);
		var costOld = 0;
		X = np.hstack([np.ones([X.length, 1]), X]);
		y = np.reshape(y, [nRows, 1]);
		if (!this._W) {
			this._W = np.random.random([np.shape(X)[1], 1]);
		}
		return { costOld, y, X };
	}

	_runEpoch(X, y) {
		var end, batchX, batchY, batchPreds, batchGrad;
		for (let start = 0; start < y.length; start += this._b) {
			end = start + this._b;
			batchX = X.slice(start, end);
			batchY = y.slice(start, end);
			batchPreds = this.evaluate(batchX);
			batchGrad = this._grad(batchX, batchY, batchPreds);
			// TODO add nesterov update
			this._update(batchGrad, (this._b > 1) ? this._b : y.length);
		}
		return [batchY, batchPreds, batchGrad];
	}

	_converged(costOld, costCurrent, stopThreshold, batchGrad) {
		return !Math.abs(parseInt((costOld - costCurrent) / stopThreshold))
			|| !parseInt(np.linalg.norm(batchGrad) / stopThreshold);
	}

	/**
	 * 
	 * @param {NDArray} X 
	 * @param {Array|NDArray} y 
	 * @param {number} maxIter 
	 * @param {number} stopThreshold 
	 * @returns 
	 */
	async fit(X, y, maxIter = 1024, stopThreshold = 1e-6) {
		// TODO flag to tell fit is done
		this.fitSync(X, y, maxIter, stopThreshold);
		return this;
	}
}

class LinearRegression {
	constructor() {

	}

	fit(X, y) {

	}

	predict(X) {

	}
}

module.exports = { GradientDescent, LinearRegression };