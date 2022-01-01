#! /usr/bin/node
"use strict";

const tf = require("@tensorflow/tfjs");
const dfd = require("danfojs-node");
const np = require("./numpy.js");
const { ARIMA } = require("./arima");

dfd.read_csv("./Data/daily-total-female-births-in-cal.csv").then(df => {
	return df["births"].tensor.array();
}).then(data => {
	data = np.array(data.slice(0, -15));
	data = np.reshape(data, [-1, 1])
	var model = ARIMA(2, 0, 2, 1e-5);
	return model.fit(data);
}).then(model => {
	return model.forecast(15);
}).then(preds => {
	console.log(preds);
	return dfd.read_csv("./Data/daily-total-female-births-in-cal.csv");
}).then(df => {
	return df["births"].tensor.array();
}).then(data => {
	return data.slice(-15);
}).then(console.log);
