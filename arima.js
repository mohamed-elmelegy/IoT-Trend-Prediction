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
        // TODO extend to include other q elements
        this._W = np.random.random([p + 1, 1]);
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

    _buildFeatures(series) {
        let X = np.diff(series, this._d);
        let labels = X.slice(this._p);
        let features = [];//[np.ones([labels.length])];
        for (let idx = 1; idx <= this._p; idx++) {
            features.push(X.slice(this._p - idx, -idx));
        }
        // TODO extend to include q elements
        features = np.vstack(features);
        features = np.transpose(features);
        return [features, labels];
    }

    _calculatePeriods(series) {
        // TODO to be implemented
        return series.length;
    }

    score(X, y) {
        // TODO to be implemented
        throw new Error("Not Implemented yet!");
    }

    fitSync(X, maxIter = 1024, stopThreshold = 1e-6) {
        // TODO extend to include q terms
        let [feats, labels] = this._buildFeatures(X);
        this._lags = labels.slice(-this._p);
        super.fitSync(feats, labels, maxIter, stopThreshold);
    }

    async fit(X, maxIter = 1024, stopThreshold = 1e-6) {
        this.fitSync(X, maxIter, stopThreshold);
        return this;
    }

    predictSync(periods) {
        // TODO the output must be consistent w/ input
        let res = this._lags.slice();
        for (let i = 0; i < periods; i++) {
            let X = res.slice(-this._p);
            X.unshift(1);
            let y = super.evaluate([X]);
            res.push(y);
        }
        return res.slice(-periods);
    }

    async predict(periods) {
        return this.predictSync(periods);
    }
}

module.exports = {
    ARIMA:
        (p, d, q, learningRate = .001, KWArgs = {}) => {
            return new AutoRegressionIntegratedMovingAverage(p, d, q, learningRate, KWArgs)
        },
}
