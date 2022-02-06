#! /usr/bin/env node
const tf = require("@tensorflow/tfjs");
const dfd = require("danfojs-node");
const { ARIMA, arima } = require("./Code/pureJS/arima");
const np = require("./Code/pureJS/numpy");
const tfArima = require("./Code/tensorflowJS/arima")

/**
 * understanding differencing & cumsum (cumulative sum)
 */
{
	x = [1, 4, 9, 16, 25];
	console.log(x);
	y = np.diff(x);
	console.log(y);
	// y = [3, 5, 7, 9];
	z = np.cumsum([x[0], ...y]);
	// [1, 3, 5, 7, 9]
	console.log(z);
	// z = [1, 4, 9, 16, 25]
}

/**
 * Understanding overloaded .at method
 */
{
	var x = np.arange(4);
	console.log("x", x);
	var toe = np.linalg.toeplitz(x);
	console.log("toe", toe);
	console.log(toe.at([0, 2]));
	console.log(toe.at([0, 2], [2]));
}

/**
 * Understanding lagging of a 1D vector
 */
{
	var x = np.arange(4);
	var n0 = arima._buildNLags(x);
	console.log("0", n0);
	var n1 = arima._buildNLags(x, 1);
	console.log("1", n1);
	var n2 = arima._buildNLags(x, 2);
	console.log("2", n2);
}

const FILE_PATH = "./Data/daily-total-female-births-in-cal.csv";

var df;
dfd
	.read_csv(FILE_PATH).then(data => {
		df = data.tensor.arraySync();
		df = np.array(df).T[1];
		df = df.map(el => parseInt(el));
		// FIXME ARIMA(0, 0, 0) is called random walk, but not implemented
		// TODO try different models of different (p, d, q) orders
		var model = ARIMA([1, 0, 0]);
		// TODO now try fitStat with as little as 64 for maxIter; it still works
		return model.fit(df.slice(0, -10), 8192, 1e-9);
	})
	.then(model => {
		console.log(model);
		var forecasted = model.forecastSync(10);
		console.log(`true:\n${df.slice(-10)}\n\nforecast:\n${forecasted}`);
		return model.update(df.slice(-10));
	})
	.then(console.log);

// TODO tf demo
const DEFAULT_PARAMS = {
	epochs: 2048,
	shuffle: false, // time series is ordered
	validationSplit: .2, // cross validation 20% test -> validation score 
	callbacks: tf.callbacks.earlyStopping({
		monitor: 'val_loss',
		patience: 3
	})
};
var tfModel;
dfd
	.read_csv(FILE_PATH).then(data => {
		df = data.tensor.arraySync();
		df = np.array(df).T[1];
		df = df.map(el => parseInt(el));
		// FIXME ARIMA(0, 0, 0) is called random walk, but not implemented
		// TODO try different models of different (p, d, q) orders
		tfModel = tfArima.ARIMA(1, 0, 0);
		// TODO now try fitStat with as little as 64 for maxIter; it still works
		return tfModel.fit(df.slice(0, -10));
	})
	.then(_ => {
		console.log(tfModel);
		var forecasted = tfModel.predictSync(10);
		console.log(`true:\n${df.slice(-10)}\n\nforecast:\n${forecasted}`);
		// TF has no update, it is different with NN (research fine tuning/transfer learning)
		// return history.update(df.slice(-10));
	})
	// .then(console.log);
