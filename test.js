#! /usr/bin/node
"use strict";

const njs = require("./arima.js");

console.log(typeof (njs));
console.dir(njs);
let x = njs.array([2, 3]);
x = njs.ones(4);
// console.log(x);
// x= njs.zeros([3, 3]);
// console.log(x);
console.log(x, njs.diag(x, 1));

x = njs.array([[3, 1], [1, 2]]);


// console.log(x, x.add(1));

console.log(x, njs.diag(x));