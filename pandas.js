#! /usr/bin/node
/**
 * TODO fit the file for tensorflow
 */
"use strict";

const np = require("./numpy");
// const tf = require("@tensorflow/tfjs-node-gpu");
const SECOND = 1000;
const MINUTE = 60 * SECOND;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;
const WEEK = 7 * DAY;
// TODO MONTH_END, MONTH_START, ANNUAL

class Series extends np.NDArray { // could be replaced by tf.tensor1d
	_dtype = undefined;
	_index = null;
	_name = null;

	constructor(data = null, index = null, dtype = null, name = null) {
		if (dtype == "datetime") {
			super(...toDateTime(data));
		} else {
			super(...data);
		}
		this._index = index;
		this._dtype = dtype;
		this._name = name;
	}

	get dt() {
		if (this.dtype === "datetime") {
			// TODO using tf.Tensor is different
			// 
			return TimeSeries.from(this);
		} else {
			throw Error("dt available only for datetime values");
		}
	}

	get dtype() {
		if (this.length &&
			((this._dtype === null) || (this._dtype === undefined))) {
			var el = this[0];
			switch (typeof (el)) {
				case "object":
					if (el instanceof Date) {
						this._dtype = "datetime";
					}
				// FIXME for additional possible data types
				// } else if (false) {}
				default:
					this._dtype = typeof (el);
			}
		}
		return this._dtype;
	}

	get index() {
		return this._index;
	}

	set index(arg) {
		if (arg.length != this.length) {
			throw Error("Index length mismatch");
		}
		this._index = arg;
	}

	asFreq(
		frequency,
		method = null,
		how = null,
		normalise = false,
		fill_value = null
	) {
		if (!(this._index instanceof TimeSeries)) {
			throw Error("Invalid operation");
		}
		if (this._index.freq == frequency) {
			return this;
		}
		// TODO have the frequency take effect
		let res = this.slice();
		switch (frequency) {
			case "L":
				break;
			case "S":
				break;
			case "T":
				break;
			case "H":
				break;
			case "D":
				break;
			case "W":
				break;
			default:
		}
		return res;
	}
}

/**
 * 
 */
class TimeSeries extends Series {
	_freq = null; // freq of sensors 

	get freq() {
		// TODO calculate frequency
		let periods = this.slice(1).sub(this.slice(0, -1));
		switch (periods / (this.length - 1)) {
			case 1:
				this._freq = "L";
				break;
			case SECOND:
				this._freq = "S";
				break;
			case MINUTE:
				this._freq = "T";
				break;
			case HOUR:
				this._freq = "H";
				break;
			case DAY:
				this._freq = "D";
				break;
			case WEEK:
				this._freq = "W";
				break;
			default:
				this._freq = undefined;
		}
		return this._freq;
	}

	get year() {
		return Series.from(this.map(el =>
			el.getFullYear()
		));
	}

	get weekday() {
		return Series.from(this.map(el =>
			el.getDay()
		));
	}

	get hour() {
		return Series.from(this.map(el =>
			el.getHours()
		));
	}

	get minute() {
		return Series.from(this.map(el =>
			el.getMinutes()
		));
	}

	get month() {
		return Series.from(this.map(el =>
			el.getMonth() + 1
		));
	}

	get second() {
		return Series.from(this.map(el =>
			el.getSeconds()
		));
	}

	get day() {
		return Series.from(this.map(el =>
			parseInt(el.toISOString().slice(8, 10))
		));
	}

	// get time() {
	// 	return Series.from(this.map(el =>
	// 		el.getTime()
	// 	));
	// }

	ceil(frequency) {
		// TODO fill the ceil
	}

	floor(frequency) {
		// TODO fill the floor
	}

	round(frequency) {
		// TODO fill the round method
	}
}

class DataFrame extends Object {
	_dtypes = null;
	_index = null;

	constructor(args) {
		switch (typeof (args)) {
			case "object":
				super(args);
				for (const key in this) {
					if (!(this[key] instanceof Series)) {
						this[key] = Series.from(this[key]);
					}
				}
				// TODO the values must match dimension, throw error if not
				this.index = Array.from(Object.values(args)[0].keys());
				break;
			case "undefined":
				super();
				break;
			default:
				super(args);
		}
	}

	get dtypes() {
		// FIXME match each column name to its data type
		if ((this._dtypes === null) || (this._dtypes === undefined)) {
			this._dtypes = Object.values(this).map(el =>
				el.dtype
			);
		}
		return this._dtypes;
	}

	get columns() {
		return Object.keys(this);
	}

	set columns(args) {
		// TODO validate the given, search override keys of object
		var keys = Object.keys(this);
		// TODO test me
		for (let i = 0; i < keys.length; i++) {
			const el = keys[i];
			this[args[i]] = this[el];
			delete (this[el]);
		}
	}

	get index() {
		return this._index;
	}

	set index(arg) {
		// TODO validate the argument index
		this._index = arg;
	}

	get values() {
		return np.array(Object.values(this)
			.map(el =>
				Array.from(el)
			));
	}

	setIndex(idx, inPlace = False) {
		var self = (inPlace) ? this : new DataFrame(this);
		if (typeof (idx) === "string") {
			// TODO search for given `idx` in columns, if not found raise an error
		} else if (idx instanceof Array) {
			if (idx.length < this.columns.length) {
				self._index = [];
				for (key in idx) {
					self._index.push(self[key]);
					delete (self[key]);
				}
			}
			if (this.index.length != idx.length) {
				throw Error("Index must be of the same length as the records number");
			}
		}
	}
}

/**
 * FIXME not really needed
 * @param {string|Array} arg 
 * @returns 
 */
function toDateTime(arg) {
	if (arg instanceof Array) {
		return arg.map(el =>
			new Date(el)
		);
	}
	else if (typeof (arg) === "string") {
		return new Date(arg);
	}
}

DataFrame.prototype.setIndex = function (arg) {
	// TODO to make it possible to use DateTime indices
}

module.exports = {
	toDateTime,
	// Series: function (arr) { return Series.from(arr); },
	DataFrame,
	Series
}
