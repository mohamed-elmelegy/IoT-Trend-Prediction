/* 
========================================== TensorFlow JS ========================================== 

    When importing TensorFlow.js from '@tensorflow/tfjs-node', the module that you get will be accelerated 
    by the TensorFlow C binary and run on the CPU. TensorFlow on the CPU uses hardware acceleration to 
    accelerate the linear algebra computation under the hood.

    This package will work on Linux, Windows, and Mac platforms where TensorFlow is supported.

    Notes:-
    1) You do not have to import '@tensorflow/tfjs' or add it to your package.json. This is indirectly 
    imported by the node library.
    2) In case '@tensorflow/tfjs-node' doesn't work, You have to import '@tensorflow/tfjs'
    
========================================== TensorFlow JS ========================================== 
*/
const tf = require('@tensorflow/tfjs')

/* 
    For TF API docs visit this -----> https://js.tensorflow.org/api/latest/
*/

v1 = tf.tensor1d([1, 2, 3, 4]);
v2 = tf.tensor1d([10, 20, 30, 40]);
m1 = tf.tensor2d([[1, 2], [3, 4]]);
m2 = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
num = tf.scalar(5);

// Adding Two Vectors
function add(v1, v2) {
    return tf.add(v1, v2);
}
console.log("Addition: ")
add(v1, v2).print();

// Multiplying Two Vectors
function multiply(v1, v2) {
    return tf.mul(v1, v2);
}
console.log("\nMultiplication: ")
multiply(v1, v2).print();

// Square Vector
function square(v1) {
    return tf.square(v1);
}
console.log("\nSquare: ")
square(v1, v2).print();

// Dot Product
function dotP(m1, m2) {
    return tf.dot(m1, m2);
}
console.log("\nDot Product: ")
dotP(m1, m2).print();

