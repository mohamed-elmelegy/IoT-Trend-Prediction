#! /usr/bin/node
"use strict";

// const tf = require("@tensorflow/tfjs");
const dfd = require("danfojs-node");
const np = require("./numpy.js");
const { ARIMA } = require("./arima");

dfd.read_csv("./Data/daily-total-female-births-in-cal.csv").then(df => {
	return df["births"].tensor.array();
}).then(data => {
	data = np.array(data.slice(0, -10));
	var model = ARIMA([1, 1, 1], { learningRate: 1e-5 });
	return model.fit(data);
}).then(model => {
	return model.forecast(10);
}).then(preds => {
	console.log(preds);
	return dfd.read_csv("./Data/daily-total-female-births-in-cal.csv");
}).then(df => {
	return df["births"].tensor.array();
}).then(data => {
	return data.slice(-10);
}).then(console.log);
