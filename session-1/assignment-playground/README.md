# Axiom — Learning OS: Session 1 Interactive Playground

This project folder contains a self-contained, interactive web application that implements, visualizes, and proves the four core assignment claims of Session 1.

*   **App Location**: [index.html](file:///Users/utkarshupadhyay/Personal/Projects/era-v5/session-1/assignment-playground/index.html)
*   **Documentation Location**: [README.md](file:///Users/utkarshupadhyay/Personal/Projects/era-v5/session-1/assignment-playground/README.md)

---

## 🛠️ Project Architecture

To ensure zero dependencies, fast page loads, and ease of deployment, the application is written as a self-contained single-page application (SPA). It implements a lightweight, fully functional **Neural Network engine from scratch in JavaScript**.

### The Custom JS Neural Network Engine
The engine is written in vanilla JS to perform forward propagation, analytical backpropagation, and weights updates:

1.  **Dense Layer (`DenseLayer`)**:
    *   **Initialization**: Employs **Xavier initialization** for linear and sigmoid layers, and **He initialization** ($\sigma = \sqrt{2/d_{in}}$) for ReLU layers to prevent vanishing/exploding gradients.
    *   **Forward Pass**: Computes $z = x \cdot W + b$, then applies the activation:
        *   Sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$
        *   ReLU: $\max(0, z)$
        *   Linear: $z$
    *   **Backward Pass**: Computes analytical gradients:
        *   Pre-activation error $\delta = \frac{dL}{dz} = \frac{dL}{da} \odot f'(z)$
        *   Gradients: $dW = x^T \delta$ and $db = \sum \delta$
        *   Gradient w.r.t layer input: $d_{in} = \delta W^T$ (passed back to the previous layer)
2.  **Sequential Wrapper (`SequentialNetwork`)**:
    *   Chains dense layers together, piping the forward output of layer $l$ as the input to layer $l+1$, and propagating gradients in reverse order during the backward pass.
3.  **Loss Function**:
    *   Uses **Binary Cross-Entropy (BCE)** for classification:
        $$L = -\frac{1}{B} \sum_{i=1}^B \left( y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right)$$
    *   Includes clamping of predictions ($\hat{y} \in [10^{-7}, 1 - 10^{-7}]$) to prevent numerical underflow and `NaN` errors during log calculations.

---

## 🔍 Detailed Analysis of the 4 Assignment Claims

### 1. S1-1 · Activations exist for a reason
*   **Claim**: A model with no nonlinearity can only draw a straight boundary, failing to separate concentric rings; adding one ReLU hidden layer solves it.
*   **Mathematical Explanation**:
    A single linear layer (with a sigmoid output) defines the decision boundary at $z = w_1 x_1 + w_2 x_2 + b = 0$. This is mathematically a straight line. Since concentric rings require a circular or closed boundary to separate, a straight line is geometrically constrained to cutting the circles in half, capping the accuracy at **~50-55%**.
    
    By adding a hidden layer of size $H$ with a **ReLU activation**, the network projects the 2D input into an $H$-dimensional space and folds it along the ReLU hinges ($z > 0$). The final output layer draws a linear hyperplane in this folded space, which projects back to a piecewise-linear, closed boundary in the original 2D space.
*   **UI Implementation**: 
    Watch a Linear model and a ReLU hidden-layer model train side-by-side on 300 concentric ring points. The background displays classification confidence (fading to neutral at the boundary, saturated red/blue for high confidence). The linear boundary remains a flat line while the ReLU boundary bends to wrap around the inner ring, reaching **~99%+ accuracy**.

### 2. S1-2 · Depth without nonlinearity is a lie
*   **Claim**: 5 stacked linear layers mathematically collapse into a single linear map, so a 5-layer linear net is no stronger than a 1-layer net; both fail the ring task. Inserting ReLUs between the 5 layers solves it.
*   **Mathematical Collapse Proof**:
    Let the outputs of a 5-layer linear network be:
    $$y_1 = x W_1 + b_1$$
    $$y_2 = y_1 W_2 + b_2 = (x W_1 + b_1) W_2 + b_2 = x (W_1 W_2) + (b_1 W_2 + b_2)$$
    
    Repeating this substitution through all 5 layers expands the final output (pre-sigmoid) to:
    $$y_5 = x (W_1 W_2 W_3 W_4 W_5) + b_1 W_2 W_3 W_4 W_5 + b_2 W_3 W_4 W_5 + b_3 W_4 W_5 + b_4 W_5 + b_5$$
    
    We define the collapsed, effective parameters:
    *   **Effective Weight Matrix**: $W_{eff} = W_1 \times W_2 \times W_3 \times W_4 \times W_5$ (collapses to size $2 \times 1$)
    *   **Effective Bias**: $b_{eff} = (((b_1 W_2 + b_2) W_3 + b_3) W_4 + b_4) W_5 + b_5$ (collapses to a scalar)
    
    Hence, $y_5 = x \cdot W_{eff} + b_{eff}$. This is structurally identical to a single-layer linear model.
*   **UI Implementation**:
    *   Compares 1-Layer Linear, 5-Layer Linear, and 5-Layer ReLU boundaries side-by-side. Both linear boundaries remain identical straight lines.
    *   **Mathematical Collapse Dashboard**: Computes $W_{eff}$ and $b_{eff}$ in real-time. It feeds a sample coordinate $[0.5, 0.5]$ through the actual 5-layer network and compares the result to the direct $x \cdot W_{eff} + b_{eff}$ formula. The output difference is displayed as `0.00000000`, proving the mathematical collapse.
    *   Includes a collapsible **"How the Collapse Math Works"** panel detailing the algebraic proof.

### 3. S1-3 · Embeddings learn similarity from next-token predictions
*   **Claim**: Trained only to predict the next token in a tiny synthetic grammar, the embedding table clusters related tokens automatically, even though similarity was never supplied.
*   **Grammar Rules**:
    *   `[Animal] eat [Fruit] <EOS>`
    *   `[Animal] chase [Animal] <EOS>`
    *   `[Animal] see [Fruit/Animal] <EOS>`
    *   `<EOS> -> [Animal]` (to start the next sentence)
*   **Clustering Principle**:
    Because the network only looks at the current token $x_t$ to predict $x_{t+1}$ using a softmax layer:
    $$p = \text{softmax}(e \cdot W_{out} + b_{out})$$
    
    Words that share the same next-token targets must output the same probability distribution. To minimize cross-entropy loss, the model must map these words to the same (or very close) embedding vectors $e$.
    
    *   `cat`, `dog`, and `cow` all predict Verbs next, so their embeddings cluster together.
    *   `apple` and `mango` both predict `<EOS>` next, so they cluster together.
    *   **Why `<EOS>` was introduced**: Initially, `chase` and `apple`/`mango` both predicted Animals next, causing a coordinate overlap. By introducing `<EOS>`, `apple`/`mango` predict `<EOS>`, while `chase` predicts Animals, and `<EOS>` transitions to `cat`/`dog` (excluding `cow` to make `<EOS>` and `chase` targets distinct). This successfully separates the nouns, verbs, and structural tokens into distinct quadrants.
*   **Smooth Drift (BGD + Decay)**:
    Instead of noisy Stochastic Gradient Descent, the model trains using **Batch Gradient Descent**, averaging the gradients over the entire corpus. A **Learning Rate Decay** factor reduces the step size over time, allowing words to glide smoothly and settle in their final positions with zero jitter.
*   **UI Implementation**: 
    A 2D scatter plot showing words drifting into distinct color-coded clusters. Hovering or clicking a word badge displays a live nearest-neighbors table sorted by Cosine Similarity.

### 4. S1-4 · Memorization vs Generalization
*   **Claim**: A high-capacity model on tiny data drives train loss to ~0 while test loss remains high; growing the dataset closes this gap.
*   **The Over-parameterization Gap**:
    A network with layers `[2, 64, 32, 1]` contains thousands of trainable weights. 
    *   When trained on $N=20$ points, the model has far more degrees of freedom than data constraints. It memorizes the coordinates by drawing a highly complex, squiggly boundary that loops around individual noisy points (100% train accuracy, 0 train loss). However, it fails completely on unseen test data, resulting in a **large generalization gap**.
    *   When trained on $N=2000$ points, the data constraints restrict the model from overfitting to individual noise. It is forced to learn the global, smooth shape of the spiral, bringing the training and testing loss curves close together.
*   **UI Implementation**:
    Allows toggling between dataset sizes $N = 20, 200, 2000$. It displays the training split boundary and testing split boundary side-by-side, along with a live Chart.js plot of Train vs. Test Loss, highlighting the shrinking generalization gap.

---

## 🚀 How to Run & Deploy

### Running Locally
To run the interactive playground on your machine:
1.  Navigate into this project directory.
2.  Start a local development server using Python:
    ```bash
    python3 -m http.server 8000
    ```
3.  Open your browser and go to:
    ```
    http://localhost:8000/index.html
    ```

### Deploying to Netlify
To make your assignment shareable:
1.  Open the Netlify drop website: [app.netlify.com/drop](https://app.netlify.com/drop)
2.  Drag and drop the `assignment-playground` folder directly into the upload area.
3.  Netlify will instantly generate a live shareable URL for your Session 1 submission!
