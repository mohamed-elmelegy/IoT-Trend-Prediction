# IoT Trend Prediction Using ARIMA Model

> `ARIMA`: stands for Auto Regression Integrated Moving Average. [Wiki Link](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)

<img src="https://i0.wp.com/www.projectguru.in/wp-content/uploads/2020/09/1-1.jpg?fit=538%2C194&ssl=1">

Here is a JavaScript implementation for the following models from `ARIMA` family:

- AR(p).
- MA(q).
- ARMA(p,q).
- ARIMA(p, d, q)

In two different ways of implementation:
- Pure JS [Full Documentation](./Code/pureJS/README.md) | [Code](./Code/pureJS/arima.js)
- TensorFlow JS [Full Documentation](./Code/tensorflowJS/README.md) | [Code](./Code/tensorflowJS/arima.js)


It provides ARIMA model APIs. The full equation for the non-seasonal arima model is:

$$
\Phi(B)(y_t'-\mu)=\Theta(B)\varepsilon_t
$$

where,

- $B$ is the backshift operator: $By_t=y_{t-1};B\varepsilon_t=\varepsilon_{t-1};B\phi=\phi;B^dy_t=y_{t-d}$.
- $y_t'$ is the differenced series: $y_t'=y_t-y_{t-1}$
- $\mu$ is the mean of the differenced series: $\mu=\frac{1}{T}\sum_{t=0}^Ty_t'$.
- $\varepsilon_t$ is the residual: $\varepsilon_t=y_t-\hat{y_t}$.
- $\Phi$ are the auto regression weights: $\Phi(B)=(1-B\phi_1-...-B^p\phi_p)$.
- $\Theta$ are the moving average weights: $\Theta(B)=(1+B\theta_t+...+B^q\theta_q)$.

<br>

It supports working on [Master of Things (MoT)](http://www.masterofthings.com) platform for serving IoT solutions.

<img src="https://images.wuzzuf-data.net/files/company_covers/thumbs/9476685373d323b29752206014247a63.jpeg">


---

## **Credits**

This project was carried out under the supervision of:
- [Information Technology Institute (ITI)](https://www.iti.gov.eg/) 
- [SpimeSenseLabs](http://www.spimesenselabs.com/) Company


<img alt="ITI Logo" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTWDMFZ4fFvJV7D94Kt6dyYqen2t8I2f9WTymVVDKlSoJDzoMCkQb4ZjLrLx5XYbF-kl-c&usqp=CAU" width="300" height="93"> <img alt="SpimeSenseLabs Logo" src="https://images.wuzzuf-data.net/files/company_logo/SpimeSenseLabs-Egypt-8358.jpeg">
 
, as a graduation project from [9-month professional training program in Artificial Intelligence](https://ai.iti.gov.eg/epita/ai-engineer/) powerded by the [Egyptian Ministry of Communications and Information Technology (MCIT)](https://www.mcit.gov.eg/) in cooperation with [EPITA](https://www.epita.fr/en/) in France.

<img alt="EPITA Logo" src="https://ai.iti.gov.eg/wp-content/uploads/2020/08/logo-epita.png">


### **Team**
- Hossam Khairullah. [LinkedIn](https://www.linkedin.com/in/hossam-khir-allah/)
- Mohamed Ashraf El-Melegy. [LinkedIn](https://www.linkedin.com/in/mohamedashrafelmelegy/)
- Mostafa Mohamed Fathy. [LinkedIn](https://www.linkedin.com/in/mostafamohamedfathy/)