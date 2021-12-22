const tf = require('@tensorflow/tfjs')

function train(x, y) {
    // mean of our inputs and outputs
    const x_mean = x.mean().dataSync()[0];
    const y_mean = y.mean().dataSync()[0];

    //total number of values
    const n = x.shape[0]

    // using the formula to calculate the b1 and b0
    numerator = 0
    denominator = 0

    for (let i = 0; i < n; i++) {
        // console.log(x.dataSync()[i]);
        numerator += (x.dataSync()[i] - x_mean) * (y.dataSync()[i] - y_mean)
        denominator += (x.dataSync()[i] - x_mean) ** 2
    }
    b1 = numerator / denominator
    b0 = y_mean - (b1 * x_mean)
    return [b0,b1]
}

function predict(x,weigths) {
    return weigths[0] + weigths[1] * x;
}


//Example
// initializing our inputs and outputs
const x = tf.tensor1d([...Array(20)].map((_, i) => i));
const y = tf.tensor1d([...Array(20)].map((_, i) => i * 3));


weigths=train(x,y);
y_pred=predict(10,weigths);

console.log(y_pred);
