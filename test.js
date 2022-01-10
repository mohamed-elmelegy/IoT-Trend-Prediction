#! /usr/bin/node
"use strict";

const tf = require("@tensorflow/tfjs");
const dfd = require("danfojs-node");
const np = require("./numpy.js");
const { ARIMA } = require("./arima");

var x;
dfd.read_csv("./Data/daily-total-female-births-in-cal.csv").then(df => {
	return df["births"].tensor.array();
}).then(data => {
	data = np.array(data.slice(0, -10));
	x = data.slice(-10);
	var model = ARIMA([1, 1, 1], { learningRate: 1e-5 });
	return model.fit(data);
}).then(model => {
	console.log(model.metrics);
	return model.forecast(10);
}).then(preds => {
	[x, preds] = [tf.tensor(x), tf.tensor(preds)];
	const mae = tf.metrics.meanAbsoluteError(x, preds).arraySync();
	const mape = tf.metrics.meanAbsolutePercentageError(x, preds).arraySync();
	const mse = tf.metrics.meanSquaredError(x, preds).arraySync();
	const rmse = Math.sqrt(mse);
	const nrmse = rmse / x.mean().arraySync();
	console.log(`mae:\t${mae}`);
	console.log(`mape:\t${mape}`);
	console.log(`mse:\t${mse}`);
	console.log(`rmse:\t${rmse}`);
	console.log(`nrmse:\t${nrmse}`);
}).catch(console.error);
