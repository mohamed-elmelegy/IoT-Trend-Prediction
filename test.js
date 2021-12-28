#! /usr/bin/node
"use strict";

const np = require("./numpy.js");
const pd = require("./pandas");
const { GradientDescent } = require("./linreg");
// const { model } = require("@tensorflow/tfjs-node-gpu");

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
// console.log(dataList);
var x = {};
dataList.forEach(el => {
	for (var [key, val] of Object.entries(el)) {
		// console.log(`${key}:${val}`);
		x[key] = (x[key] == undefined) ? [val] : [...x[key], val];
	}
});
console.log(x);
x = np.linspace(0, 20);
let y = x.mul(-1).add(2);
x = np.reshape(x, [x.length, 1]);
let mod = new GradientDescent();
mod = mod.fit(x, y, 4096);
mod.then((res) => {
	console.log(res._W);
	let i = np.array([21]);
	return res.predict(i);
}).then(console.log);
