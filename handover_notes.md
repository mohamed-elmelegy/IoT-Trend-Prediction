# Handover Notes

| ![Univariate time series](https://otexts.com/fpp3/fpp_files/figure-html/stationary-1.png) |
| :---------------------------------------------------------------------------------------: |
|                 all series are univariate, only `b` & `g` are stationary                  |

## Auto Regression (AR) Integrated (I) Moving Average (MA) `ARIMA`

The simplest idea is that either the lags or the lagged residuals or both contribute to the current value

---

### Differencing

Working with the **change in the value** rather than the actual value. It is the simple operation of subtracting each element from the previous element. In doing so, the 1st element gets discarded as it wouldn't be subtracted from something else.

$
Y'=(y_t-y_{t-1})|_{t=1}^{t=T}
$

Important property of this operation is that it is recursive in nature; meaning that to make a differencing of order 2, you do a differencing on the **actual vector**, then apply differencing again but this time on the resulting **differenced vector**. That would map to working on the **change in the _change_ of the value**.

#### Backshift notation

$
B^dy_t=y_{t-d}
$

the backshift operator has no effect on non-time bound variables.

$
B^d\phi_i=\phi_i
$

---

### Auto Regression

The current value, (or the next value in case of forecasting), is a _linear combination_ of the last `p` lags.

$
y_t=\phi_1y_{t-1}+\phi_2y_{t-2}+...+\phi_py_{t-p}+\varepsilon_t
$

rearranging:

$
\phi_0y_t-\phi_1y_{t-1}-\phi_2y_{t-2}-...-\phi_py_{t-p}=\varepsilon_t
\\
(1-\phi_1B-\phi_2B^2-...-\phi_pB^p)y_t=\varepsilon_t
\\
\Phi(B)y_t=\varepsilon_t
$

---

### Moving Average

The current value, (or the next value in case of forecasting), is a _weighted moving average_ of the last `q` lagged residuals.

$
y_t=\varepsilon_t+\theta_1\varepsilon_{t-1}+\theta_2\varepsilon_{t-2}+...+\theta_q\varepsilon_{t-q}
$

rearranging

$
y_t=(1+\theta_1B+\theta_2B^2+...+\theta_qB^q)\varepsilon_t
\\
y_t=\Theta(B)\varepsilon_t
$

---

### ARMA & ARIMA

The ARMA is the combination of `AR` & `MA` models.

$
\Phi(B)y_t=\Theta(B)\varepsilon_t
$

The ARIMA simply considers the differencing before applying `ARMA`.

$
\Phi(B)y'_t=\Theta(B)\varepsilon_t
\\
y'_t=(1-B)^dy_t
$

> In coding, that maps to differencing the vector and apply the model, then reverse the differencing with cumulative sum.

The above equations are in linear form, leading to `linear regression`.

## Intuition behind Linear Regression

a line equation would be in the form of: $y=wx+b$; where `w` represents the _slope_, `b` represents the _bias_.

In higher dimensions a line becomes a hyperplane: $y=w_1x_1+...+w_ix_i+b$.

When we have multiple records, then it is time for matrices. The matrix form of the hyperplane (by extension also the line) becomes: $Y=W^TX+b$; where `b` here becomes a vector.

Solving the matrix equation analytically requires inverting the matrix, and that has certain conditions that aren't always met. So we rely on numerical methods to solve it. There are few popular methods, we are currently focused on `Gradient Descent`.

### Gradient Descent

| ![Gradient Descent](https://miro.medium.com/max/600/1*iNPHcCxIvcm7RwkRaMTx1g.jpeg) |
| :--------------------------------------------------------------------------------: |
|                                  Gradient Descent                                  |

The simple intuition for this algorithm:

1. assume random weights (slopes from the equation `W` & bias `b`).
2. substitute in the equation to obtain predictions $\hat{Y}$.
3. using a pre-defined **cost function** to quantise the difference between $Y$ and $\hat{Y}$, find the cost.
4. minimise the cost, by updating the assumptions about the weights using the gradient.
5. go to step 2 and repeat until convergence (the error is within an acceptable rate &epsilon;, defined for each case).

What is a gradient you might ask? it is the derivative of the cost function with respect to the weights & bias.

> Basically, in this particular case, it is a vector of the partial derivative of the cost function w.r.t $w_1,...,w_i$ & `b`.
>
> for the cost functions, theoretically, there can be infinite number of them, but most common are mean squared error `MSE`, & mean absolute error `MAE`.

MSE:

$$
J=\frac{1}{2n}\sum_{i=0}^n(y_i-\hat{y_i})^2
$$

update equations:

$$
W=W-\alpha\frac{\partial{J}}{\partial{W}}
\\
=W-\frac{\alpha}{n}\sum_{i=0}^n(-X(y-\hat{y}_i)
$$

$$
b=b-\alpha\frac{\partial{J}}{\partial{b}}
\\
=b-\frac{\alpha}{n}\sum_{i=0}^n(\hat{y}_i-y)
$$

where &alpha; is the learning rate, a way to tune the size of the step taken towards the solution.

> There are variants of the gradient descent \[vanilla, mini-batch, stochastic\], which alters the value of `n` in the update equation.
>
> `GD` can be improved by introducing the momentum. Simply to encourage large learning steps when the cost is _high_, and small learning steps when the cost is _relatively small_ \[RMSProp, Adagrad, Adam\]
>
> This algorithm is sometimes referred to by the name `perceptron algorithm`. In layman terms, a perceptron can easily be transformed to a neuron, by having an activation function after summation. In other words, a perceptron is a neuron which has the linear activation function `y=x`.

---

## Implementing the ARIMA model

The linear regression is used, where the features $x_1,...,x_i$ are replaced by $y_{t-1},...,y_{t-p}$ and/or $\varepsilon_1,...,\varepsilon_{t-q}$, and the `b` is dropped, and $W=\begin{bmatrix}\Phi(B) \\ \Theta(B)\end{bmatrix}$

> Due to the nature of `ARIMA` being built on top of lags, the -ve sign in the update equation becomes +ve (see [code]("./../Code/pureJS/arima.js#L22"));

By definition, &epsilon; should be noise around the mean of the series, but in order to _fit_ the model, it takes another mathematical meaning: $\varepsilon_t=y_t-\hat{y_t}$

---

## Correction of the Log Likelihood formula

<!-- $\sigma^2=\frac{1}{T}\sum_i^T\varepsilon_i^2$ -->

$
LL=-\frac{1}{2}(T\ln(2\pi\sigma^2)+\frac{1}{\sigma^2}\sum_{t=0}^T\varepsilon_t)
$
