#! /usr/bin/node
"use strict";

const tf = require('@tensorflow/tfjs');


function extractIdxTensor1D(listOfObjects) {
	var res = [];
	var idx = [];
	for (let i = 0; i < listOfObjects.length; i++) {
		let obj = listOfObjects[i];
		for (const key in obj) {
			if (key === "T") {
				idx.push(obj[key]);
			} else {
				res.push(obj[key]);
			}
		}
	}
	res = tf.tensor1d(res);
	return [idx, res];
}

function extractTensor2D(listOfObjects) {
	var res = [];
	var idx = [];
	for (let i = 0; i < listOfObjects.length; i++) {
		let obj = listOfObjects[i];
		for (const key in obj) {
			if (key === "T") {
				idx.push(obj[key]);
			} else {
				res.push(obj[key]);
			}
		}
	}
	return tf.tensor2d([idx, res]);
}

function diff(tensor, ord = 1) {
	if (ord == 0) {
		return tensor;
	}
	let self = tf.clone(tensor);
	for (let o = 0; o < ord; o++) {
		let other = self.slice(0, self.size - 1);
		self = self.slice(1);
		self = tf.sub(self, other);
	}
	return self;
}

module.exports = {
	extractIdxTensor1D, extractTensor2D
}