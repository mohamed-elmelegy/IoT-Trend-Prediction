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
]