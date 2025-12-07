# Optimization

# First order optimization methods in ML (Gradient based)

Why it's we need to bother about it at all !

If it convince you we need to think about what best we can do

Below are few most important algoritms used by ML

# Basic Algorithm

### SGD (Stochastic Gradient Descent)

- This (Pure SGD) is classic algorithm and now a days it's rarely used
- Stochastic stand for the randomness in data point, Let's consider we have N training sample and we choose m randomly and trained network for that samples. A complete pass through the whole training set called as _epoch_ (Model have seen all N example one time)

- Key psuedo structure

```
                    N → datapoints
                    Initialize weight vector → w

                    weight error → e

                    Set learning rate → l
                    n ← n + 1
                    while not converge:
                        w ← w - l * e
                        n ← (n+1)% N

                    return weight w
```

```
                x += - learning_rate * dx
```

```
pytorch uses →  optim = torch.optim.SGD(model.parameters(), lr=3e-4)
```

### Momentum

- Momentum is borrowed from physics and it works same here too
- mu \* velocity → friction, mu value here is between [0, 1] usually 0.5 or
  0.99. This compenent takes care of damping in steep direction so the oscillations can be controlled. While as in shallow direction the velocity accumulate consistently and shows nice effect and speed

```
        take n sample indexed from 1...N
        Batch size B
        error function per mini batch E(w)
        learning rate l
        momentum coeff. mu
        initial weight w

        n ← 1
        velocity ← 0     # this is momentum term

        while not converge:

            grad ← gradient of E(w) on batch[n : n+B-1]

            velocity ← mu * velocity - l * grad
            w ← w + velocity

            n ← n + B
            if n > N:
                shuffle data
                n ← 1

        return w


```

```
pytorch uses →

optim = torch.optim.SGD(params, lr=0.001, momentum=0.9, dampening=0, weight_decay=0, nesterov=False, *, maximize=False, foreach=None, differentiable=False, fused=None)

```

We are really rolling it pretty fast now, let's see if there are any way to do even better

# Adaptive Algorithm

These algorithms allocate separate learning rate for parameters and automatically adapt learning rates throughout learning. This instantly gives edge over basic algorithms.

General idea - Steep gradient have high derivative and when it plugged in denominator (inverse scaling), it penalize (decrese in this case) the learning rate and vice versa.

### AdaGrad (Adaptive Gradient) update

AdaGrad (Adaptive Gradient) is directly borrowed from convex optimization area.

```
            learning rate l
            cache = 0                  # accumulator for squared gradients

            while not converge:

                grad = gradient of E(w)

                cache = cache + grad**2
                w = w - l * grad / (sqrt(cache) + 1e-7)

            return w

```

The gradient accumulated to cache and more the gradient accumulation quicker the decay in learning rate.

Problem with AdaGrad is over long time the cache get accumulated leading to stop learning.

### RMSprop (Root Mean Square Propagation)

To address AdaGrad problem which suite Non convex setup. The cache is converted to leaky cache and eventually it just remebers recent history and discard the older.

```
        learning rate l
        cache = 0
        decay rate d   # d → 0.9
        while not converge :
            grad = gradient of E(w)
            cache += d * cache + (1 - d) * grad**2   # Leaky cache
            w += - l * grad / (np.sqrt(cache) + 1e-7)  # 1e-7 is small stabilizing constant to avoid divide by zero error
            # This scale learning rate dynamically per paramter

        return w

```

### Adam (Adaptive + Moments)

```
        learning rate l
        decay rate d1 , d2  # set to default d1 → 0.9 , d2 → 0.995/0.999
        m , v = 0
        for t in xrange(0, big_num) :
            grad = gradient of E(w)
            m = d1 * m + (1 - d1) * grad           # momentum
            v = d2 * v + (1 - d2) * (grad**2)      # RMSprop
            m /= 1 - d1**t  # in-line update → correct bias → compension as we initialized to zero
            v /= 1 - d2**t  # in-line update →  correct bias → compension as we initialized to zero
            w -= l * m / (np.sqrt(v) + 1e-7)  # 1e-7 is small stabilizing constant to avoid divide by zero error

        return w

```

### AdamW (Adaptive + Moments + decouple Weight decay)

To avoid overfitting in AdamW the weight decay process is decoupled from gradient update. This also leads to better convergence than Adam.

```
        learning rate l
        decay rate d1 , d2  # set to default d1 → 0.9 , d2 → 0.995/0.999
        m , v = 0
        weight decay rho
        for t in xrange(0, big_num) :
            grad = gradient of E(w)
            m = d1 * m + (1 - d1) * grad           # momentum
            v = d2 * v + (1 - d2) * (grad**2)      # RMSprop
            m /= 1 - d1**t  # in-line update → correct bias → compension as we initialized to zero
            v /= 1 - d2**t  # in-line update → correct bias → compension as we initialized to zero
            w -= l * m / (np.sqrt(v) + 1e-7)  # 1e-7 is small stabilizing constant to avoid divide by zero error
            w -= l * rho * w   # Apply decoupled weight decay

```

### Learning rate decay

Broad idea - you can't just select just 'very high' / 'good' / 'low' learning rate throughout training rather we need to switch it depending on stage of learing. Example we can start with hig learning later it eventually start to get slow so we have to update our learning rate.

Below few option we can explore to dynamically decay rate

- Step Decay - Decay learning rate by few epoch with defined function e.g. divide by 2
- Exponential decay - using exponent decay the rate (empirically proved)
- 1/t decay - divide the rate with (1 + kt) factor

```
Preference →

Batch Gradient → SGD → SGD with Momemtum → Nesterov Momentum → AdaGrad → RMSprop → Adam → AdamW

Note → Currently most llm`s are trained using AdadW optimizer

```

# Second order optimization methods in ML

This optimization is Gradient + Hessian

- Second order Taylor expansion
- Newton Method
- Quasi-Newton Method (BGFS)
- L-BFGS

Although it converge faster and have no hyperparameters, it's impractial for Deep learning as the Hessian matrix is order of o(n\*\_2) e.g. if we have billion parameter Hessian will be billion \* billion followed by inversion which is very computational heavy.

_Reference_

_1 - Deep Learning - Christopher and Huge Bishop_

_2 - Deep Learning - Ian Goodfellow_

_3 - Convex Optimization - Stephen Boyd_

_4 - [CS231n Winter 2016: Lecture 6: Neural Networks Part 3 / Intro to ConvNets](https://www.youtube.com/watch?v=hd_KFJ5ktUc&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&index=6)_

_5 - [AdamW](https://optimization.cbe.cornell.edu/index.php?title=AdamW)_

Thank you for seeing this through to the end 🙌

<div style="display: flex; justify-content: space-between; width: 100%; font-size: 1.1em;">
  <a href="/notes" class="page-link">Notes</a>
  <a href="/" class="page-link">Home</a>
</div>
