#! /usr/bin/node
"use strict";

const np = require("./numpy.js");
// const pd = require("./pandas");
// const { GradientDescent } = require("./linreg");
const { ARIMA } = require("./arima");


/**
 * TODO the query on MoT returns a list of objects, so it is advised to convert
 * them to a DataFrame for easier manipulation
 */

var dataList = [
	{
		"TimeStamp": new Date("1995-05-02"),
		"value": 42
	},
	{
		"TimeStamp": new Date("2021-05-02"),
		"value": 32
	}
];
var x = {};
dataList.forEach(el => {
	for (var [key, val] of Object.entries(el)) {
		// console.log(`${key}:${val}`);
		x[key] = (x[key] == undefined) ? [val] : [...x[key], val];
	}
});
var x = np.linspace(0, 20);
var y = x.slice(0, -2);
var arima = ARIMA(1, 1, 1);
arima.fit(y).then(model => {
	return model.predict(2);
}).then(console.log).catch(err => {
	console.error(err)
}).finally(() => {
	console.log(x.slice(-2));
});