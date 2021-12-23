#! /usr/bin/node
"use strict";

/**
 * naive compare array
 * https://stackoverflow.com/questions/3115982/how-to-check-if-two-arrays-are-equal-with-javascript/16430730
 * 
 * @param {Array} other 
 * @returns 
 */
Array.prototype.equalsTo = function (other) {
	return JSON.stringify(this) == JSON.stringify(other);
}

/**
 * 
 */
class NDArray extends Array {

}

/**
 * a: array-like or iterable
 * @param {Array} a 
 * @returns 
 */
function array(a) {
	return NDArray.from(a);
}

/**
 * 
 * @param {number|NDArray} size 
 * @returns 
 */
function empty(size) {
	if (typeof (size) === "number") {
		return new NDArray(size);
	}
	if (size instanceof Array) {
		let p = 1;
		p = size.flat(size.length - 1)
			.reduce((p, el) =>
				p * el
			);
		// return reshape(array(Array(p)), size);
		return reshape(Array(p), size);
	}
}

/**
 * 
 * @param {number|Array} size 
 * @returns 
 */
function zeros(size) {
	if (typeof (size) === "number") {
		size = [size]
	}
	return reshape(
		empty(size)
			.flatten()
			.fill(0),
		size);
}

/**
 * 
 * @param {number|Array} size 
 * @returns 
 */
function ones(size) {
	if (typeof (size) === "number") {
		size = [size]
	}
	return reshape(
		empty(size)
			.flatten()
			.fill(1),
		size);
}

/**
 * 
 * @param {NDArray} vector 
 * @param {number} k 
 * @returns 
 */
function diag(vector, k = 0) {
	// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Spread_syntax
	let res = [];
	switch (ndim(vector)) {
		case 1:
			res = vector.length + Math.abs(k);
			res = Array(2).fill(res);
			res = zeros(res);
			for (let i = Math.max(0, -k), j = Math.max(0, k), l = 0;
				l < vector.length;
				i++, j++, l++) {
				res[i][j] = vector[l];
			}
			return res;
		case 2:
			let lim = Math.min(...shape(vector));
			res = empty(lim);
			for (let i = 0; i < lim; i++) {
				res[i] = vector[i][i];
			}
			return res;
		default:
			throw Error("Array must be 1D or 2D");
	}
}

/**
 * create an identity matrix
 * @param {number} size 
 * @returns 
 */
function eye(size) {
	return diag(ones(size));
}

/**
 * returns vector[i+1] - vector[i]
 * @param {NDArray} vector 
 * @param {number} order 
 * @returns 
 */
function diff(vector, order = 1) {
	if (order == 0) {
		return vector;
	}
	let self = array(vector);
	for (let d = 0; d < order; d++) {
		let other = self.slice(0, -1);
		self = self.slice(1);
		self = self.map((s, idx) =>
			s - other[idx]
		);
	}
	return self;
}

/**
 * number of dimensions of the array
 * @param {Array|NDArray} vector 
 * @returns 
 */
function ndim(vector) {
	let dim = 0;
	let self = [...vector];
	for (dim = 0; self instanceof Array; dim++) {
		self = self[0]; // FIXME array elements are not required to be the same here
	}
	return dim;
}

/**
 * FIXME transposing only 2D
 * https://stackoverflow.com/questions/17428587/transposing-a-2d-array-in-javascript
 * @param {NDArray} vector 
 * @returns 
 */
function transpose(vector) {
	const dim = ndim(vector);
	if (dim == 1) {
		return vector;
	}
	if (dim == 2) {
		// return vector[0].map((_, j) =>
		// 	vector.map((row) => row[j])
		// );
		return array(vector[0].map((_, j) =>
			[...vector].map((row) => row[j])
		));
	}
}

/**
 * https://stackoverflow.com/questions/10237615/get-size-of-dimensions-in-array
 * @param {Array|NDArray} vector 
 * @returns 
 */
function shape(vector) {
	let self = [...vector];
	const n = ndim(vector);
	let shape = [vector.length]
	for (let dim = 1; dim < n; dim++) {
		shape.push(self[0].length);
		self = self[0];
	}
	return shape;
}

/**
 * reshapes the array into the given shape, if possible
 * @param {Array|NDArray} vector 
 * @param {Array} size 
 * @returns 
 */
function reshape(vector, size) {
	if (typeof (size) === "number") {
		size = [size];
	}
	if (size.map(el =>
		el == -1
	).reduce((tot, el) =>
		tot + el
	) > 1) {
		throw Error("Cannot infer more than one dimension");
	}
	let tSize = 1;
	tSize = shape(vector)
		.reduce((tSize, el) =>
			tSize * el
		);
	let oSize = 1;
	oSize = size.reduce((oSize, el) =>
		oSize * el
	);
	if (oSize < 0) {
		switch (tSize % oSize) {
			case 0:
				let index = size.indexOf(-1);
				size[index] = -tSize / oSize;
				break;
			default:
				throw Error("Unable to infer missing dimension");
		}
	} else if (tSize != oSize) {
		throw Error("Incompatible shapes");
	}
	// vector = array(vector).flatten();
	vector = array(vector).flatten();
	// FIXME keeping the largest array as ndarray, & internal arrays 
	// as normal arrays
	vector = [...vector];
	let result = [];
	size = size.reverse();
	for (let idx = 0; idx < size.length - 1; idx++) {
		let step = size[idx];
		for (let i = 0; i < vector.length; i += step) {
			result.push(vector.slice(i, i + step));
		}
		vector = result;
		result = [];
	}
	// FIXME had to restore the original order for size
	size = size.reverse();
	return array(vector);
}

/**
 * TODO continue if needed
 * https://numpy.org/doc/stable/reference/generated/numpy.sum.html
 * https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/Reduce
 * @param {NDArray} vector 
 * @param {number} axis 
 * @param {number} initialValue 
 * @returns 
 */
function sum(vector, axis = null, initialValue = 0) {
	if ((ndim(vector) == 1) || axis === null) {
		// full array or 1D array
		return vector.flatten()
			.reduce(((sum, el) =>
				sum + el
			),
				initialValue);
	}
	// FIXME to avoid breaking
	return vector;
	// if ((axis < 0) && (-axis <= this.length)) {
	// 	axis = this.length - axis
	// }
}

/**
 * FIXME implement higher dimensions
 * @param {NDArray} a 
 * @param {NDArray} b 
 * @returns 
 */
function dot(a, b) {
	const shapeA = shape(a);
	const shapeB = shape(b);
	if (shapeA[shapeA.length - 1] != shapeB[0]) {
		throw Error("Internal dimension mismatch");
	}
	// vector dot product
	if ((ndim(a) == 1) && (ndim(b) == 1)) {
		return sum(a.mul(b));
	}
	if (ndim(a) == 1) {
		a = transpose([a]);
	}
	if (ndim(b) == 1) {
		b = transpose([b]);
	}
	// FIXME 2D operation only
	b = transpose(b);
	return a.map(row =>
		b.map(col =>
			sum(array(row).mul(col))
		)
	);
}

/**
 * check if given shapes allow for broadcasting
 * @param {Array} a 
 * @param {Array} b 
 * @returns 
 */
function canBroadcast(a, b) {
	let [i, j] = [[...a].reverse(), [...b].reverse()];
	[i, j] = (i.length >= j.length) ? [i, j] : [j, i];
	var res = true;
	j.map((el, idx) =>
		res &= (el == i[idx]) | (el == 1) | (i[idx] == 1)
	);
	return res;
}

/**
 * 
 * @param {Array} size0 
 * @param {Array} size1 
 * @returns 
 */
function resultantShape(size0, size1) {
	matchDimensions(size0, size1);
	return size0.map((el, i) =>
		Math.max(el, size1[i])
	);
}

/**
 * 
 * @param {Array} a 
 * @param {Array} b 
 */
function matchDimensions(a, b) {
	var size = a.length - b.length;
	if (size < 0) {
		a.unshift(...ones(-size));
	} else if (size > 0) {
		b.unshift(...ones(size));
	}
}

/**
 * https://numpy.org/doc/stable/user/basics.broadcasting.html
 * @param {NDArray} vector 
 * @param {Array} size 
 * @returns 
 */
function broadcast(vector, size) {
	let vSize = shape(vector);
	if (!canBroadcast(vSize, size)) {
		throw Error("Can not broadcast");
	}
	matchDimensions(vSize, size);
	vector = vector.flatten();
	var tempSize = [];
	for (let i = size.length - 1; i >= 0; i--) {
		if (vSize[i] == size[i]) {
			tempSize.shift();
			tempSize.unshift(-1, size[i]);
			vector = reshape(vector, tempSize);
		} else {
			vector = vector.map(el =>
				new NDArray(size[i]).fill(el)
			);
			tempSize = shape(vector);
		}
	}
	return reshape(vector, size);
}

/**
 * https://stackoverflow.com/questions/3895478/does-javascript-have-a-method-like-range-to-generate-a-range-within-the-supp
 * @param {number} start 
 * @param {number} end 
 * @param {number} step 
 * @returns 
 */
function arange(start, end, step) {
	if (end === undefined) {
		end = start;
		start = 0;
	}
	step = (step === undefined) ? 1 : step;
	if ((end < start) && (step > 0)) {
		return [];
	}
	[start, end] = (start < end) ? [start, end] : [end, start];
	let res = array(Array(end).keys())
		.slice(start)
		.filter(el =>
			!((el - start) % step)
		);
	return (step >= 0) ? res : res.reverse();
}


/**
 * 
 * @param {number} start 
 * @param {number} stop 
 * @param {number} num 
 * @returns 
 */
function linspace(start, stop, num = 50) {
	let step = (stop - start) / (num - 1);
	let res = [];
	for (let element = start; element < stop; element += step) {
		// https://stackoverflow.com/questions/2221167/javascript-formatting-a-rounded-number-to-n-decimals
		res.push(parseFloat(element.toFixed(8)));
	}
	return array(res);
}

/**
 * FIXME works for 2D arrays only
 * @param {Array} elements 
 * @returns 
 */
function vstack(elements) {
	let res = [];
	let size = shape(elements[0]);
	if (size.length == 1) {
		elements.forEach(el => res.push(el));
	} else {
		elements.forEach(el => res.push(...el));
	}
	return array(res);
}

/**
 * FIXME works for 2D arrays only
 * @param {Array} elements 
 * @returns 
 */
function hstack(elements) {
	// FIXME edge case
	var res;
	if (ndim(elements[0]) == 1) {
		res = [];
		elements.forEach(el => {
			res.push(...el)
		});
		return array(res);
	}
	res = elements.map(el => transpose(el));
	return transpose(vstack(res));
}

/**
 * https://stackoverflow.com/questions/7135874/element-wise-operations-in-javascript
 * @param {NDArray|number} other 
 * @param {callbackfn} op 
 * @returns 
 */
NDArray.prototype.iOperation = function (other, op) {
	let sThis = shape(this);
	if (typeof (other) === "number") {
		return reshape(
			this.flatten()
				.map(el =>
					op(el, other)
				),
			sThis
		);
	}
	other = array(other);
	let sOther = shape(other);
	if (sThis.equalsTo(sOther)) {
		other = other.flatten();
		return reshape(
			this.flatten()
				.map((el, i) =>
					op(el, other[i])
				),
			sThis
		);
	} else {
		// TODO handling broadcasting
		sThis = resultantShape(sThis, sOther);
		other = broadcast(other, sThis).flatten();
	}
	return reshape(
		broadcast(this, sThis).flatten()
			.map((el, i) =>
				op(el, other[i])
			),
		sThis);
}

/**
 * 
 * @param {NDArray} other 
 * @returns 
 */
NDArray.prototype.add = function (other) {
	return this.iOperation(other, (a, b) => a + b);
}

/**
 * 
 * @param {NDArray} other 
 * @returns 
 */
NDArray.prototype.mul = function (other) {
	return this.iOperation(other, (a, b) => a * b);
}

/**
 * 
 * @param {NDArray} other 
 * @returns 
 */
NDArray.prototype.sub = function (other) {
	return this.iOperation(other, (a, b) => a - b);
}

/**
 * 
 * @param {NDArray} other 
 * @returns 
 */
NDArray.prototype.div = function (other) {
	return this.iOperation(other, (a, b) => a / b);
}

/**
 * 
 * @param {NDArray} other 
 * @returns 
 */
NDArray.prototype.equals = function (other) {
	return this.iOperation(other, (a, b) => a == b);
}

/**
 * 
 * @param {NDArray} other 
 * @returns 
 */
NDArray.prototype.power = function (other) {
	return this.iOperation(other, (a, b) => a ** b);
}

/**
 * 
 * @returns 
 */
NDArray.prototype.flatten = function () {
	return this.flat(ndim(this) - 1);
}

const linalg = {
	norm: function (vector, ord = 2) {
		// FIXME edge cases
		if (ord === Infinity) {

		} else if (ord === -Infinity) {

		} else if (ord == 0) {

		}
		return sum(vector.power(ord)) ** 1 / ord
	}
}

const random = {
	random: function (size) {
		return reshape(empty(size).flatten().map(_ => Math.random()), size);
	}
}

module.exports = {
	array, empty, diff, dot, ndim, reshape, shape, sum, transpose, diag, ones, zeros, eye, arange, vstack, hstack, NDArray, linalg, linspace, random
}
