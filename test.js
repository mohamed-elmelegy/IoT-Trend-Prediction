#! /usr/bin/node
"use strict";

const np = require("./numpy.js");
const pd = require("./pandas");
// const LinearRegression = require("./linreg");

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
var y = new pd.DataFrame(x);
// console.log(y.dtypes);
y.forEach(el => {
    console.log(el);
});