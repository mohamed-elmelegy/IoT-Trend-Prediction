#! /usr/bin/node
/**
 * TODO update the file to match tensorflow
 */
"use strict";

const np = require("./numpy");
// const tf = require("@tensorflow/tfjs-node-gpu");

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
				// let error = y.sub(yHat);
				let error = yHat.sub(y);
				return np.dot(np.transpose(X), error);
			}
		}
		// TODO nesterov update
		this._update = (kwargs["nesterov"]) ? this.updateNesterov : function (gradient, m, vt1 = 0) {
			// this._W -= this.vt(gradient, m, vt1);
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
		// this._gamma * vt1 + 
		return gradient.mul(this._alpha).div(m).add(this._gamma * vt1);
	}

	/**
	 * 
	 * @param {NDArray} X 
	 * @returns 
	 */
	async predict(X) {
		return await Promise.resolve(this.evaluate(X));
	}

	fitSync(X, y, maxIter = 1024, stopThreshold = 1e-6) {
		// let ut = 0 // TODO support adaptive grad
		let n = y.length;
		this._b = (this._b) ? this._b : n;
		let costOld = this._costFn(y.slice(-this._b), np.zeros([this._b]), this._b);
		X = np.hstack([np.ones([X.length, 1]), X]);
		y = np.reshape(y, [n, 1]);
		if (!this._W) {
			this._W = np.random.random([np.shape(X)[1], 1]);
		}
		for (let epoch = 0; epoch < maxIter; epoch++) {
			for (let start = 0; start < n; start += this._b) {
				let end = start + this._b;
				let batchX = X.slice(start, end);
				let batchY = y.slice(start, end);
				let batchPreds = this.evaluate(batchX);
				let batchGrad = this._grad(batchX, batchY, batchPreds);
				// TODO add nesterov update
				this._update(batchGrad, (this._b > 1) ? this._b : n); // TODO use ut for adaptive
				let costCurrent = this._costFn(batchY, batchPreds, this._b);
				if (!Math.abs(parseInt((costOld - costCurrent) / stopThreshold))
					|| !parseInt(np.linalg.norm(batchGrad) / stopThreshold)
				) {
					return;
				} else {
					costOld = costCurrent;
				}
			}
		}
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
		await this.fitSync(X, y, maxIter, stopThreshold);
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