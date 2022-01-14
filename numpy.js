#! /usr/bin/node
"use strict";

/**
 * naive compare array
 * https://stackoverflow.com/questions/3115982/how-to-check-if-two-arrays-are-equal-with-javascript/16430730
 * 
 * @param {Array} that 
 * @returns 
 */
Array.prototype.equalsTo = function (that) {
	return JSON.stringify(this) == JSON.stringify(that);
}

/**
 * 
 */
class NDArray extends Array {
	get T() {
		return transpose(this);
	}

	get shape() {
		return shape(this);
	}

	get ndim() {
		return ndim(this);
	}

	get strides() {
		var size = this.shape.slice(1);
		size.push(1);
		for (let i = size.length - 2; i >= 0; i--) {
			size[i] *= size[i + 1];
		}
		return size;
	}

	/**
	 * 
	 * @param  {...any} args 
	 * @returns 
	 */
	at(...args) {
		if (args.length > this.ndim) {
			throw new Error("Index out of bound");
		}
		switch (ndim(args)) {
			case 1:
				// var res = this.slice();
				var res = [...this.slice()];
				args.forEach(idx => {
					// res = res[idx];
					res = res.at(idx);
				});
				return res;
			default:
				var res = [];
				if (args.length > 1) {
					res = this.at(args[0]);
					args = args.slice(1);
					args.forEach(idx => {
						res = res.map(el => [...array(el).at(idx)]);
					});
					return res;
				}
				[args] = args;
				args.forEach(i => {
					res.push(super.at(i));
				});
				return array(res);
		}
	}
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
	// let self = array(vector);
	let self = [...vector];
	for (let d = 0; d < order; d++) {
		let other = self.slice(0, -1);
		self = self.slice(1);
		self = self.map((s, idx) =>
			s - other[idx]
		);
	}
	// return self;
	return array(self);
}

/**
 * TODO does not work with axis
 * @param {Array|NDArray} vector 
 * @param {number} axis
 * @returns 
 */
function cumsum(vector, axis = null) {
	var total = 0;
	return vector.map((el) => total += el);
}

/**
 * TODO axis works only for 2D
 * @param {Array|NDArray} vector
 * @param {number} axis 
 * @returns 
 */
function mean(vector, axis = null) {
	if (axis != null) {
		// TODO supporting only 2D
		// return sum(array(vector), axis).div(vector[axis].length);
		return sum(array(vector), axis).div(vector.shape[axis]);
	}
	return sum(array(vector)) / prod(vector.shape);
}

/**
 * 
 * @param {Array|NDArray} vector 
 * @returns 
 */
function std(vector, axis = null) {
	var mu = mean(vector, axis);
	if (axis != null) {
		// TODO supports only 2D
		mu = reshape(mu, [-1, 1]);
		return sum(
			array(vector).sub(mu).power(2),
			axis
		).div(vector.shape[axis]).power(.5);
	}
	const sigma2 = sum(array(vector).sub(mu).power(2)) / prod(vector.shape);
	return Math.sqrt(sigma2);
}

/**
 * 
 * @param {Array|NDArray} vector 
 * @param {number} axis 
 * @returns 
 */
function prod(vector, axis = null) {
	vector = array(vector);
	if (axis != null) {
		// TODO support higher dimension prod
	}
	return vector.flatten().reduce((p, el) => p * el, 1);
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
	vector = array(vector);
	var self = vector.flatten();
	var res = [];
	var temp = [];
	var steps = vector.strides.slice(0, -1);
	steps.forEach(step => {
		for (let o = 0; o < step; o++) {
			for (let i = o; i < self.length; i += step) {
				temp.push(self[i]);
			}
			res.push(temp);
			temp = [];
		}
		[self, res] = [res, []];
	});
	return array(self);
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
	// FIXME keeping the largest array as ndarray, & internal arrays 
	// as normal arrays
	vector = [...array(vector).flatten()];
	// vector = [...vector];
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
	vector = array(vector);
	if ((axis === null) || (ndim(vector) == 1)) {
		// full array or 1D array
		return vector.flatten()
			.reduce(((sum, el) =>
				sum + el
			),
				initialValue);
	}
	// TODO supporting 2D sum
	vector = (axis) ? vector : transpose(vector);
	return vector.map(el => sum(el));
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
	[a, b] = [array(a), array(b)];
	const shapeA = a.shape;
	const shapeB = b.shape;
	// const shapeA = shape(a);
	// const shapeB = shape(b);
	if (shapeA.at(-1) != shapeB[0]) {
		throw Error("Internal dimension mismatch");
	}
	// vector dot product
	if ((a.ndim == 1) && (b.ndim == 1)) {
		return sum(a.mul(b));
	}
	if (a.ndim == 1) {
		a = a.reshape([1, -1]);
	}
	if (b.ndim == 1) {
		b = b.reshape([-1, 1]);
	}
	var resShape = [...shapeA.slice(0, -1), ...shapeB.slice(1)];
	var res = a.reshape([-1, b.shape[0]]);
	b = transpose(b);
	res = res.map(row =>
		b.map(col =>
			sum(array(row).mul(col))
		)
	);
	return reshape(res, resShape);
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
	vector = array(vector);
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
	step = step || 1;
	if ((end < start) && (step > 0)) {
		return [];
	}
	[start, end] = (start < end) ? [start, end] : [end, start];
	var res = array(Array(end).keys())
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
	elements.forEach(el => {
		el = (el.ndim == 1) ? el : [...el];
		res.push(...el);
	});
	return array(res);
}

/**
 * FIXME works for 2D arrays only
 * @param {Array} elements 
 * @returns 
 */
function hstack(elements) {
	var res;
	// FIXME edge case
	if (elements.every(el => ndim(el) == 1)) {
		res = [];
		elements.forEach(el => {
			res.push(...[...el])
		});
		return array(res);
	}
	res = elements.map(el => transpose(el));
	return transpose(vstack(res));
}

/**
 * 
 * @param {CallbackFn} op 
 * @param  {...any} args 
 * @returns 
 */
NDArray.prototype.apply = function (op, ...args) {
	const size = shape(this);
	if (args.length) {
		var n = args.reduce((m, el) =>
			m + (typeof (el) === "number")
			, 0);
		if (n == args.length) {
			return reshape(this.flatten().map(el =>
				op(el, ...args)
			), size);
		}
		args = transpose(args.map(el =>
			broadcast((el instanceof Array) ? el : [el], size).flatten()
		));
		return reshape(this.flatten().map((el, i) => op(el, ...args[i])), size);
	} else {
		return reshape(this.flatten().map(el => op(el)), size);
	}
}

/**
 * https://stackoverflow.com/questions/7135874/element-wise-operations-in-javascript
 * @param {NDArray|number} that 
 * @param {callbackfn} op 
 * @returns 
 */
NDArray.prototype.iOperation = function (that, op) {
	let shapeThis = shape(this);
	if (typeof (that) === "number") {
		return reshape(
			this.flatten()
				.map(el =>
					op(el, that)
				),
			shapeThis
		);
	}
	that = array(that);
	let shapeThat = shape(that);
	if (shapeThis.equalsTo(shapeThat)) {
		that = that.flatten();
		return reshape(
			this.flatten()
				.map((el, i) =>
					op(el, that[i])
				),
			shapeThis
		);
	}
	// handling broadcasting
	shapeThis = resultantShape(shapeThis, shapeThat);
	that = broadcast(that, shapeThis).flatten();
	return reshape(
		broadcast(this, shapeThis).flatten()
			.map((el, i) =>
				op(el, that[i])
			),
		shapeThis);
}

/**
 * 
 * @param {NDArray} that 
 * @returns 
 */
NDArray.prototype.add = function (that) {
	return this.apply((a, b) => a + b, that);
}

/**
 * 
 * @param {NDArray} that 
 * @returns 
 */
NDArray.prototype.mul = function (that) {
	return this.apply((a, b) => a * b, that);
}

/**
 * 
 * @param {NDArray} that 
 * @returns 
 */
NDArray.prototype.sub = function (that) {
	return this.apply((a, b) => a - b, that);
}

/**
 * 
 * @param {NDArray} that 
 * @returns 
 */
NDArray.prototype.div = function (that) {
	return this.apply((a, b) => a / b, that);
}

/**
 * 
 * @param {NDArray} that 
 * @returns 
 */
NDArray.prototype.equals = function (that) {
	return this.apply((a, b) => a == b, that);
}

/**
 * 
 * @param {NDArray} that 
 * @returns 
 */
NDArray.prototype.power = function (that) {
	return this.apply((a, b) => a ** b, that);
}

/**
 * 
 * @returns 
 */
NDArray.prototype.flatten = function () {
	return this.flat(ndim(this) - 1);
}

/**
 * 
 * @returns 
 */
NDArray.prototype.toArray = function () {
	const s = this.shape;
	return [...reshape(this.flatten(), s)];
}

/**
 * 
 * @param {Array} size 
 * @returns 
 */
NDArray.prototype.reshape = function (size) {
	return reshape(this, size);
}

/**
 * 
 * @param {NDArray} that 
 * @returns 
 */
NDArray.prototype.dot = function (that) {
	return dot(this, that);
}

/**
 * 	
 * @returns 
 */
NDArray.prototype.sum = function (axis = null, initialValue = 0) {
	return sum(this, axis, initialValue);
}

/**
 * 
 * @returns 
 */
NDArray.prototype.mean = function (axis = null) {
	return mean(this, axis);
}

/**
 * 
 * @returns 
 */
NDArray.prototype.std = function (axis = null) {
	return std(this, axis);
}

/**
 * 
 * @param {number} axis 
 * @returns 
 */
NDArray.prototype.prod = function (axis = null) {
	return prod(this, axis);
}

const linalg = {
	/**
	 * 
	 * @param {NDArray} vector 
	 * @param {number} ord 
	 * @returns 
	 */
	norm: function (vector, ord = 2) {
		// FIXME edge cases
		if (ord === Infinity) {

		} else if (ord === -Infinity) {

		} else if (ord == 0) {

		}
		return sum(vector.power(ord)) ** (1 / ord)
	},
	/**
	 * 
	 * @param {Array|NDArray} c 
	 * @param {Array|NDArray} r 
	 * @returns 
	 */
	toeplitz: function (c, r = null) {
		if (r) {
			// TODO handle row not null
		} else {
			r = [...c].reverse();
			var l = c.length;
			return c.map((_, i) =>
				[...r.slice(-1 - i), ...c.slice(1, l - i)]
			);
		}
	}
}

const random = {
	/**
	 * 
	 * @param {Array} size 
	 * @returns 
	 */
	random: function (size) {
		if (!size) {
			return Math.random();
		}
		size = (typeof (size) === 'number') ? [size] : size;
		return reshape(empty(size).flatten()
			.map(_ =>
				Math.random()
			),
			size);
	}
}

module.exports = {
	array, empty, diff, dot, ndim, reshape, shape, sum, transpose, diag, ones,
	zeros, eye, arange, vstack, hstack, NDArray, linalg, linspace, random,
	cumsum, mean, std, prod
}
