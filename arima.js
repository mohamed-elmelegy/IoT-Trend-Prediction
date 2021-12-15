#! /usr/bin/node
"use strict";

/**
 * naive compare array
 * @param {Array} other 
 * @returns 
 */
Array.prototype.equals = function (other) {
	return JSON.stringify(this) == JSON.stringify(other);
}

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
 * @param {number|Array} size 
 * @returns 
 */
function empty(size) {
	if (typeof (size) === "number") {
		return array(Array(size));
	} if (size instanceof Array) {
		let p = 1;
		p = size.flat().reduce((p, el) => p * el);
		return reshape(array(Array(p)), size);
	}
}

function zeros(size) {
	if (typeof (size) === "number") {
		size = [size]
	}
	return reshape(empty(size).flat().fill(0), size);
}

function ones(size) {
	if (typeof (size) === "number") {
		size = [size]
	}
	return reshape(empty(size).flat().fill(1), size);
}

function diag(vector, k = 0) {
	let lim = Math.min(...shape(vector));
	let res = [];
	switch (ndim(vector)) {
		case 1:
			lim = vector.length + k;
			let size = Array(2).fill(lim);
			res = zeros(size);
			// TODO build a sparse matrix with the given diagonal
			
			// for (let i = k, j = 0; i < lim; i++) {
			// 	res[k] = vector[i - k];
			// }
			return res;
		case 2:
			// TODO extract the diagonal
			res = empty(lim);
			for (let i = 0; i < lim; i++) {
				// res.push(vector[i][i]);
				res[i] = vector[i][i];
			}
			return res;
		default:
			throw Error("Array must be 1D or 2D");

	}
}

function eye(size) {
	size = size * size;
	let res = new NDArray(size);
	// TODO return the correct result
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
	let self = vector.slice();
	for (let d = 0; d < order; d++) {
		let other = self.slice(0, -1);
		self = self.slice(1);
		self = self.map((s, idx) => s - other[idx]);
	}
	return self;
}

/**
 * number of dimensions of the array
 * @param {NDArray} vector 
 * @returns 
 */
// Array.ndim = 
function ndim(vector) {
	let dim = 0;
	let self = vector.slice();
	for (dim = 0; self instanceof Array; dim++) {
		self = self[0]; // FIXME array elements are not required to be the same here
	}
	return dim;
}

// FIXME transposing only 2d
function transpose(vector) {
	const dim = ndim(vector);
	if (dim == 1) {
		return vector;
	}
	// https://stackoverflow.com/questions/17428587/transposing-a-2d-array-in-javascript
	if (dim == 2) {
		return vector[0].map((_, j) => vector.map((row) => row[j]));
	}
}

/**
 * https://stackoverflow.com/questions/10237615/get-size-of-dimensions-in-array
 * @param {Array} vector 
 * @returns 
 */
function shape(vector) {
	let self = vector.slice();
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
 * @param {NDArray} vector 
 * @param {Array} size 
 * @returns 
 */
function reshape(vector, size) {
	let product = 1;
	// let tSize = vector.shape().reduce((product, el) => product * el);
	let tSize = shape(vector).reduce((product, el) => product * el);
	product = 1;
	let oSize = size.reduce((product, el) => product * el);
	if (tSize != oSize) {
		throw Error("Incompatible shapes");
	}
	vector = vector.flat();
	let result = new NDArray();
	size = size.reverse();
	for (let idx = 0; idx < size.length; idx++) {
		let step = size[idx];
		for (let i = 0; i < vector.length; i += step) {
			result.push(vector.slice(i, i + step))
		}
		vector = result;
		result = new NDArray();
	}
	return vector[0];
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
		// 1D array
		return vector.flat().reduce(((sum, el) => sum + el), initialValue)
	}
	// FIXME avoiding breaking
	return vector;
	// if ((axis < 0) && (-axis <= this.length)) {
	// 	axis = this.length - axis
	// }
}

// FIXME implement higher dimensions
function dot(a, b) {
	const shapeA = shape(a);
	const shapeB = shape(b);
	if (shapeA[shapeA.length - 1] != shapeB[0]) {
		throw Error("Internal dimension mismatch");
	}
	// vector dot product
	if ((ndim(a) == 1) && (ndim(b) == 1)) {
		return sum(a.mul1d(b));
	}
	if (ndim(b) == 1) {
		b = transpose([b]);
	}
	// FIXME 2D operation only
	b = transpose(b);
	return a.map(row => b.map(col => sum(row.mul(col))));
}

NDArray.prototype.iOperation = function (op) {

}

// FIXME naive broadcasting method
// NDArray.broadcast = function (element, shape) {
// 	shape = shape.reverse();
// 	for (let idx = 0; idx < shape.length; idx++) {
// 		element = NDArray(shape[idx]).fill(element);
// 	}
// 	return element;
// }

/**
 * element-wise multiplication for 1D array
 * @param {NDArray} other 
 * @returns 
 */
NDArray.prototype.mul1d = function (other) {
	if (typeof (other) === 'number') {
		// other = Array.coerce(other, this.shape());
		other = NDArray(this.length).fill(other);
	}
	if (this.length != other.length) {
		throw new Error("Dimension mismatch");
	}
	return this.map((el, idx) => el * other[idx]);
}

/**
 * element-wise addition for 1D array
 * @param {NDArray} other 
 * @returns 
 */
NDArray.prototype.add1d = function (other) {
	if (typeof (other) === 'number') {
		// other = Array.coerce(other, this.shape());
		other = new NDArray(this.length).fill(other);
	}
	if ((this.length != other.length) || (ndim(this) != ndim(other))) {
		throw new Error("Dimension mismatch");
	}
	return this.map((el, idx) => el + other[idx]);
}


/**
 * element-wise subtraction for 1D array
 * @param {NDArray} other 
 * @returns 
 */
NDArray.prototype.sub1d = function (other) {
	return this.add1d(other.mul1d(-1));
}

// TODO element-wise division if needed


// FIXME adding higher dimensions w/ recursion
NDArray.prototype.add = function (other) {
	let sThis = shape(this);
	if (typeof (other) === "number") {
		return reshape(this.flat().map(el => el + other), sThis);
	}
	let sOther = shape(other);
	if (JSON.stringify(sThis) == JSON.stringify(sOther)) {
		other = other.flat();
		return reshape(this.flat().map((el, i) => el + other[i]), sThis);
	}
	// TODO handling broadcasting
	// if ((ndim(this) == 1) && ((typeof (other) === "number") || (ndim(other) == 1))) {
	// 	return this.add1d(other);
	// }
	// FIXME avoiding breaking
	return this;
}

// FIXME multiply higher dimension
NDArray.prototype.mul = function (other) {
	if ((ndim(this) == 1) && ((typeof (other) === "number") || (ndim(other) == 1))) {
		return this.mul1d(other);
	}
	// FIXME avoiding breaking
	return this;
}

// FIXME debug code
// let a = array([1, 1, 2]);
// a = a.add(1);
// console.log(a);
// a = a.add(a);
// console.log(a);
// let b = diff(a)
// let c = array([[a], [a.map((el) => el * 2), b]]);
// let d = array([a, a,]);
// let e = array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]);
// d = dot(d, transpose(d))
// console.log(shape(d));
// d = reshape(d, [4, 1])
// console.log(shape(d));
// let x = empty([5, 2]);

module.exports = {
	array, empty, diff, dot, ndim, reshape, shape, sum, transpose, diag, ones, zeros
}
