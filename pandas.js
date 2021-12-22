#! /usr/bin/node
"use strict";
const np = require("./numpy");

// FIXME Series & DataFrame should be objects instead

class Series extends np.NDArray {
	_dtype = null;
	_index = null;

	get dt() {
		if (this.dtype === "datetime") {
			return TimeSeries.from(this);
		} else {
			throw Error("dt available only for datetime values");
		}
	}

	get dtype() {
		if ((this._dtype === null) || (this._dtype === undefined)) {
			var el = this[0];
			switch (typeof (el)) {
				case "object":
					if (el instanceof Date) {
						// return "datetime";
						this._dtype = "datetime";
					}
				// FIXME for additional possible data types
				// } else if (false) {}
				default:
					// return typeof (el);
					this._dtype = typeof (el);
			}
		}
		return this._dtype;
	}

	get index() {
		return this._index;
	}

	set index(arg) {
		// TODO validate arg
		this._index = arg;
	}
}

/**
 * 
 */
class TimeSeries extends Series {
	_freq = null; // freq of sensors 

	get freq() {
		// TODO calculate frequency
		return this._freq;
	}

	set freq(arg) {
		this._freq = arg;
		// TODO have the frequency take effect
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
	toDateTime, Series: function (arr) { return Series.from(arr); }, DataFrame
}
