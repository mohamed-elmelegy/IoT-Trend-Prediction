function DataListAddAsync(name, value) {
    return new Promise((resolve, reject) => {
      const dataListObj = {
        name: name,
        value: value,
        insertAt: "tail"
      };
      DataList.add(dataListObj, dataListCallBack);
  
      function dataListCallBack(err, res) {
        if (err) {
          return reject(err);
        }
        resolve(res);
      }
    });
  }
  
  function DataListGetAsync(name) {
    return new Promise((resolve, reject) => {
      const dataListObj = {
        name: name
      }
      DataList.get(dataListObj, dataListCallBack);
  
      function dataListCallBack(err, res) {
        if (err) {
          return reject(err);
        }
        resolve(res);
      }
    });
  }
  
  function SearchInAsync(query) {
    return new Promise((resolve, reject) => {
      SearchIn(query, searchIn_callback);
  
      function searchIn_callback(err, result) {
        if (err) {
          return reject(err);
        }
  
        // write your code here
        resolve(result);
      }
    });
  }
  
  /**
  * naive compare array
  * https://stackoverflow.com/questions/3115982/how-to-check-if-two-arrays-are-equal-with-javascript/16430730
  *
  * @param {Array} other
  * @returns
  */
  Array.prototype.equalsTo = function(other) {
    return JSON.stringify(this) == JSON.stringify(other);
  }
  
  /**
  * https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/flat
  * @param {Array} arr
  * @param {number} depth
  * @returns
  */
  function flat(arr, depth = 1) {
    return depth > 0 ? arr.reduce((acc, val) =>
        acc.concat(Array.isArray(val) ? flat(val, depth - 1) :
          val), []) :
      arr.slice();
  }
  
  /**
  *
  * @param {number} depth
  * @returns
  */
  Array.prototype.flat = function(depth = 1) {
    return flat(this, depth);
  }
  
  class NDArray extends Array {
    /**
    * https://stackoverflow.com/questions/7135874/element-wise-operations-in-javascript
    * @param {NDArray|number} other
    * @param {callbackfn} op
    * @returns
    */
    iOperation(other, op) {
      let sThis = shape(this);
      if (typeof(other) === "number") {
        return reshape(
          this.flatten()
          .map(el =>
            op(el, other)
          ),
          sThis
        );
      }
      other = array(other);
      let sOther = shape(other);
      if (sThis.equalsTo(sOther)) {
        other = other.flatten();
        return reshape(
          this.flatten()
          .map((el, i) =>
            op(el, other[i])
          ),
          sThis
        );
      } else {
        // TODO handling broadcasting
        sThis = resultantShape(sThis, sOther);
        other = broadcast(other, sThis).flatten();
      }
      return reshape(
        broadcast(this, sThis).flatten()
        .map((el, i) =>
          op(el, other[i])
        ),
        sThis);
    }
  
    /**
    *
    * @param {NDArray} other
    * @returns
    */
    add(other) {
      return this.iOperation(other, (a, b) => a + b);
    }
  
    /**
    *
    * @param {NDArray} other
    * @returns
    */
    mul(other) {
      return this.iOperation(other, (a, b) => a * b);
    }
  
    /**
    *
    * @param {NDArray} other
    * @returns
    */
    sub(other) {
      return this.iOperation(other, (a, b) => a - b);
    }
  
    /**
    *
    * @param {NDArray} other
    * @returns
    */
    div(other) {
      return this.iOperation(other, (a, b) => a / b);
    }
  
    /**
    *
    * @param {NDArray} other
    * @returns
    */
    equals(other) {
      return this.iOperation(other, (a, b) => a == b);
    }
  
    /**
    *
    * @param {NDArray} other
    * @returns
    */
    power(other) {
      return this.iOperation(other, (a, b) => a ** b);
    }
  
    /**
    *
    * @returns
    */
    flatten() {
      // FIXME not supported on MoT
      return this.flat(ndim(this) - 1);
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
    if (typeof(size) === "number") {
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
    if (typeof(size) === "number") {
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
    if (typeof(size) === "number") {
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
        for (let i = Math.max(0, -k), j = Math.max(0, k), l = 0; l < vector.length; i++, j++, l++) {
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
    let self = array(vector);
    for (let d = 0; d < order; d++) {
      let other = self.slice(0, -1);
      self = self.slice(1);
      self = self.map((s, idx) =>
        s - other[idx]
      );
    }
    return self;
  }
  
  /**
  * FIXME does not work with axis
  * @param {Array|NDArray} vector
  * @returns
  */
  function cumsum(vector) {
    var total = 0;
    return vector.map((el) => total += el);
  }
  
  /**
  * FIXME does not work with axis
  * @param {Array|NDArray} vector
  * @returns
  */
  function mean(vector) {
    return sum(array(vector)) / vector.length;
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
    const dim = ndim(vector);
    if (dim == 1) {
      return vector;
    }
    if (dim == 2) {
      // return vector[0].map((_, j) =>
      // 	vector.map((row) => row[j])
      // );
      return array(vector[0].map((_, j) => [...vector].map((row) => row[j])));
    }
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
    if (typeof(size) === "number") {
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
    // vector = array(vector).flatten();
    vector = array(vector).flatten();
    // FIXME keeping the largest array as ndarray, & internal arrays
    // as normal arrays
    vector = [...vector];
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
    if ((ndim(vector) == 1) || axis === null) {
      // full array or 1D array
      return vector.flatten()
        .reduce(((sum, el) =>
            sum + el
          ),
          initialValue);
    }
    // FIXME to avoid breaking
    return vector;
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
    const shapeA = shape(a);
    const shapeB = shape(b);
    if (shapeA[shapeA.length - 1] != shapeB[0]) {
      throw Error("Internal dimension mismatch");
    }
    // vector dot product
    if ((ndim(a) == 1) && (ndim(b) == 1)) {
      return sum(a.mul(b));
    }
    if (ndim(a) == 1) {
      a = transpose([a]);
    }
    if (ndim(b) == 1) {
      b = transpose([b]);
    }
    // FIXME 2D operation only
    b = transpose(b);
    return a.map(row =>
      b.map(col =>
        sum(array(row).mul(col))
      )
    );
  }
  /**
  * check if given shapes allow for broadcasting
  * @param {Array} a
  * @param {Array} b
  * @returns
  */
  function canBroadcast(a, b) {
    let [i, j] = [Array.from(a).reverse(), Array.from(b).reverse()];
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
    step = (step === undefined) ? 1 : step;
    if ((end < start) && (step > 0)) {
      return [];
    }
    [start, end] = (start < end) ? [start, end] : [end, start];
    let res = array(Array(end).keys())
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
    let size = shape(elements[0]);
    if (size.length == 1) {
      elements.forEach(el => res.push(el));
    } else {
      elements.forEach(el => res.push(...el));
    }
    return array(res);
  }
  
  /**
  * FIXME works for 2D arrays only
  * @param {Array} elements
  * @returns
  */
  function hstack(elements) {
    // FIXME edge case
    var res;
    if (ndim(elements[0]) == 1) {
      res = [];
      elements.forEach(el => {
        res.push(...el)
      });
      return array(res);
    }
    res = elements.map(el => transpose(el));
    return transpose(vstack(res));
  }
  
  const linalg = {
    /**
    *
    * @param {NDArray} vector
    * @param {number} ord
    * @returns
    */
    norm: function(vector, ord = 2) {
      // FIXME edge cases
      if (ord === Infinity) {
  
      } else if (ord === -Infinity) {
  
      } else if (ord == 0) {
  
      }
      return sum(vector.power(ord)) ** 1 / ord
    }
  }
  
  const random = {
    /**
    *
    * @param {Array} size
    * @returns
    */
    random: function(size) {
      return reshape(empty(size).flatten()
        .map(_ =>
          Math.random()
        ),
        size);
    }
  }
  
  np = {
    array,
    empty,
    diff,
    dot,
    ndim,
    reshape,
    shape,
    sum,
    transpose,
    diag,
    ones,
    zeros,
    eye,
    arange,
    vstack,
    hstack,
    NDArray,
    linalg,
    linspace,
    random,
    cumsum,
    mean
  }
  
  class GradientDescent {
    /**
    *
    * @param {number} learningRate
    * @param {object} kwargs
    */
    constructor(learningRate = 0.001, kwargs = {}) {
      this._alpha = learningRate;
      this._W = kwargs["weights"];
      this._gamma = kwargs["momentum"] | 0;
      this._b = kwargs["batchSize"];
      this._costFn = kwargs["costFunction"];
      if (!this._costFn) {
        this._costFn = function(labels, predictions, m = null) {
          m = 2 * ((m) ? m : labels.length);
          return np.sum(labels.sub(predictions).power(2)) / m;
        };
      }
      // FIXME gradient is d[cost]/d_W, so it should be different for each cost function
      this._grad = kwargs["gradient"];
      if (!this._grad) {
        this._grad = function(X, y, yHat) {
          var error = yHat.sub(y);
          return np.dot(np.transpose(X), error);
        }
      }
      // TODO nesterov update
      this._update = (kwargs["nesterov"]) ? this.updateNesterov : function(gradient, m, vt1 = 0) {
        this._W = this._W.sub(this.vt(gradient, m, vt1));
      };
    }
  
    set alpha(learningRate) {
      this._alpha = learningRate;
    }
  
    get alpha() {
      return this._alpha;
    }
  
    set gamma(momentum) {
      this._gamma = momentum;
    }
  
    get gamma() {
      return this._gamma;
    }
  
    get _coef() {
      return this._W;
    }
  
    /**
    *
    * @param {NDArray} X
    * @returns
    */
    evaluate(X) {
      return np.dot(X, this._W);
    }
  
    updateNesterov(X, y, m, vt1) {
      // TODO implement nesterov's update
      throw Error("Method not implemented yet")
    }
  
    vt(gradient, m, vt1 = 0) {
      return gradient.mul(this._alpha)
        .div(m)
        .add(this._gamma * vt1);
    }
  
    /**
    *
    * @param {NDArray} X
    * @returns
    */
    async predict(X) {
      var features = X.slice();
      if (np.ndim(features) == 1) {
        features = np.reshape(features, [-1, 1]);
      }
      features = [np.ones([features.length, 1]), features];
      features = np.hstack(features);
      return this.evaluate(features);
    }
  
    fitSync(X, y, maxIter = 1024, stopThreshold = 1e-6) {
      // let ut = 0 // TODO support adaptive grad
      var costOld;
      ({
        costOld,
        y,
        X
      } = this._fitInit(X, y));
      for (let epoch = 0; epoch < maxIter; epoch++) {
        var {
          costCurrent,
          gradient
        } = this._runEpoch(X, y);
        if (this._converged(costOld, costCurrent, stopThreshold, gradient)) {
          break;
        } else {
          costOld = costCurrent;
        }
      }
    }
  
    _runEpoch(X, y) {
      var end, batchX, batchY, batchPreds, batchGrad;
      for (let start = 0; start < y.length; start += this._b) {
        end = start + this._b;
        batchX = X.slice(start, end);
        batchY = y.slice(start, end);
        batchPreds = this.evaluate(batchX);
        batchGrad = this._grad(batchX, batchY, batchPreds);
        // TODO add nesterov update
        this._update(batchGrad, (this._b > 1) ? this._b : y.length);
      }
      var costCurrent = this._costFn(batchY, batchPreds, this._b);
      return {
        costCurrent,
        gradient: batchGrad
      };
    }
  
    _fitInit(X, y) {
      var nRows = y.length;
      this._b = (this._b) ? this._b : nRows;
      // FIXME
      // var costOld = this._costFn(y.slice(-this._b), np.zeros([this._b]), this._b);
      var costOld = 0;
      X = np.hstack([np.ones([X.length, 1]), X]);
      y = np.reshape(y, [nRows, 1]);
      if (!this._W) {
        this._W = np.random.random([np.shape(X)[1], 1]);
      }
      return {
        costOld,
        y,
        X
      };
    }
  
    _converged(costOld, costCurrent, stopThreshold, batchGrad) {
      return !Math.abs(parseInt((costOld - costCurrent) / stopThreshold)) ||
        !parseInt(np.linalg.norm(batchGrad) / stopThreshold);
    }
  
    /**
    *
    * @param {NDArray} X
    * @param {Array|NDArray} y
    * @param {number} maxIter
    * @param {number} stopThreshold
    * @returns
    */
    async fit(X, y, maxIter = 1024, stopThreshold = 1e-6) {
      // TODO flag to tell fit is done
      this.fitSync(X, y, maxIter, stopThreshold);
      return this;
    }
  }
  
  class AutoRegressionIntegratedMovingAverage extends GradientDescent {
    /**
    *
    * @param {number} p
    * @param {number} d
    * @param {number} q
    * @param {number} learningRate
    * @param {object} KWArgs
    */
    constructor(p, d, q, learningRate = .001, KWArgs = {}) {
      super(learningRate, KWArgs);
      this._p = p;
      this._d = d;
      this._q = q;
    }
  
    get p() {
      return this._p;
    }
  
    get q() {
      return this._q;
    }
  
    get d() {
      return this._d;
    }
  
    /**
    *
    * @param {Array|NDArray} X
    * @param {number} nCols
    * @returns
    */
    _buildPredictors(X, nCols) {
      var predictors = [];
      for (let idx = 1; idx <= nCols; idx++) {
        predictors.push(...X.slice(nCols - idx, -idx));
      }
      return np.reshape(predictors, [-1, nCols]);
    }
  
    /**
    *
    * @param {Array|NDArray} X
    * @param {Array|NDArray} y
    */
    score(X, y) {
      // TODO to be implemented
      throw new Error("Not Implemented yet!");
    }
  
    /**
    * FIXME only works with AR variants [AR, ARI, ARIMA]
    * @param {Array|NDArray} X
    * @param {number} maxIter
    * @param {number} stopThreshold
    */
    fitSync(X, maxIter = 1024, stopThreshold = 1e-6) {
      this._initialValue = X.slice(-this._d - this._p, -this._p);
      var series = np.diff(X, this._d);
      var {
        labels,
        lags,
        residuals
      } = this._fitInit(series);
      var costOld = 0;
      var n = residuals.length;
      n = n ? n : lags.length;
      const ones = np.ones([lags.length, 1]);
      for (let epoch = 0; epoch < maxIter; epoch++) {
        var features = np.hstack([
          ones.slice(0, n),
          lags.slice(0, n),
          residuals
        ]);
        var {
          costCurrent,
          gradient
        } = super._runEpoch(features, labels.slice(0, n));
        features = np.hstack([ones, lags]);
        var arW = this._W.slice(0, this._p + 1);
        residuals = labels.sub(np.dot(features, arW));
        residuals = this._buildPredictors(residuals, this._q);
        if (super._converged(costOld, costCurrent, stopThreshold, gradient)) {
          break;
        } else {
          costOld = costCurrent;
        }
      }
      if (this._q) {
        this._residuals = residuals[0].slice(-this._q);
      }
    }
  
    /**
    *
    * @param {Array|NDArray} X
    * @returns
    */
    _fitInit(X) {
      let lags = this._buildPredictors(X, this._p);
      var labels = X.slice(this._p);
      this._W = np.zeros([this._p + this._q + 1, 1]);
      this._b = (this._b) ? this._b : labels.length;
      var residuals = this._buildPredictors(labels, this._q);
      this._lags = labels.slice(-this._p);
      labels = np.reshape(labels, [-1, 1]);
      return {
        labels,
        lags,
        residuals
      };
    }
  
    /**
    *
    * @param {Array|NDArray} X
    * @param {number} maxIter
    * @param {number} stopThreshold
    * @returns
    */
    async fit(X, maxIter = 1024, stopThreshold = 1e-6) {
      this.fitSync(X, maxIter, stopThreshold);
      return this;
    }
  
    /**
    *
    * @param {number} periods
    * @returns
    */
    forecastSync(periods) {
      let lags = this._lags.slice();
      var residuals = [];
      if (this._residuals) {
        residuals = this._residuals.slice();
      }
      for (let i = 0; i < periods; i++) {
        var X = lags.slice(-this._p);
        X.push(...residuals.slice(-this._q));
        X.unshift(1);
        X = np.reshape(X, [1, -1]);
        var y = super.evaluate(X).flatten();
        lags.push(...y);
        if (residuals.length) {
          residuals.push(np.mean(residuals));
        }
      }
      // the Integration step
      // https://stackoverflow.com/questions/43563241/numpy-diff-inverted-operation
      for (let d = this._d - 1; d >= 0; d--) {
        lags.unshift(this._initialValue[d]);
        lags = np.cumsum(lags);
      }
      return lags.slice(-periods);
    }
  
    /**
    *
    * @param {number} periods
    * @returns
    */
    async forecast(periods) {
      return this.forecastSync(periods);
    }
  
    updateSync(trueLags) {
      // TODO not implemented yet
      throw new Error("Not Implemented yet!");
    }
  
    async update(trueLags) {
      return this.updateSync(trueLags);
    }
  }
  
  
  
  function extractIdxTensor1D(listOfObjects) {
    var res = [];
    var idx = [];
    for (let i = 0; i < listOfObjects.length; i++) {
      let obj = listOfObjects[i];
      for (const key in obj) {
        if (key === "TimeStamp") {
          idx.push(obj[key]);
        } else {
          res.push(obj[key]);
        }
      }
    }
    // 	res = tf.tensor1d(res);
    return [idx, res];
  }
  
  console.log(GetPluginParameterValue('List 1','List captions'));
  
        // TODO predict
        DataListGetAsync('iti2021').then((model) => {
          var peridos = parseInt(GetPluginParameterValue('peridos_value', 'Selected item'));
          return model.forecast(peridos);
        }).then(predictions => {
          // TODO use predictions
          console.log(predictions)
          event.end();
        }).catch(event.error);
        
  