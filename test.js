#! /usr/bin/node
"use strict";

const tf = require("@tensorflow/tfjs");
const dfd = require("danfojs-node");
const np = require("./numpy.js");
const { ARIMA } = require("./arima");

var x;
const filePath = "./Data/daily-total-female-births-in-cal.csv";
dfd.read_csv(filePath).then(df => {
	return df["births"].tensor.array();
}).then(data => {
	data = np.array(data.slice(0, -10));
	x = data.slice(-10);
	var model = ARIMA([1, 1, 1]);
	return model.fit(data, 32, 1e-4);
}).then(model => {
	console.log(model.metrics);
	console.log(model.intercept);
	console.log(model.phi);
	console.log(model.theta);
	console.log(model.sigma2);
	return model.forecast(10);
}).then(console.log).catch(console.error);

dfd.read_csv(filePath).then(df => {
	return df["births"].tensor.array();
}).then(data => {
	data = np.array(data.slice(0, -10));
	x = data.slice(-10);
	var model = ARIMA([0, 0, 1], { learningRate: 1e-3 });
	return model.fit(data);
}).then(model => {
	console.log(model.metrics);
	console.log(model.intercept);
	console.log(model.phi);
	console.log(model.theta);
	console.log(model.sigma2);
	return model.forecast(10);
}).then(console.log).catch(console.error);

dfd.read_csv(filePath).then(df => {
	return df["births"].tensor.array();
}).then(data => {
	data = np.array(data.slice(0, -10));
	x = data.slice(-10);
	var model = ARIMA([0, 1, 1], { learningRate: 1e-3 });
	return model.fit(data);
}).then(model => {
	console.log(model.metrics);
	console.log(model.intercept);
	console.log(model.phi);
	console.log(model.theta);
	console.log(model.sigma2);
	return model.forecast(10);
}).then(console.log).catch(console.error);
