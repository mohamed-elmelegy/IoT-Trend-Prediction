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

var test = tf.tensor1d([3, 3, 0, 2]);
var d1 = diff(test);
var d2 = diff(test, 2);
console.log(d1.dataSync(), "\n---\n", d2.dataSync());
// var test = [{ "T": 1, "V": 4 }, { T: 2, V: 5 }];
// let [i, v] = convertTODO(test);
// let ten = tf.tensor2d([i, v])
// console.log(i);
// console.log("---");
// v.print();
// ten.print();