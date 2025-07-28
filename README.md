# MoCVD_Modeling and ML Learning Projects

## ML_Learnings

This directory contains a few ML programs I wrote initially while learning Machine Learning in the first 15 days before starting the `MoCVD_Modeling` project.

- My first ML model was programming the XOR function.
- Later, I developed a **cat recognition model**, both **with and without using Keras**:
  - First, I implemented it **without any ML libraries** (like TensorFlow), using only **NumPy**.
  - Then, I built a **CNN-based model using Keras** for better accuracy and efficiency.

## MoCVD_Modeling (Paper_Replication)

Hew we are aiming to to **map initial process variables**, such as **Growth Temperature**, to **intermediate material descriptors**, and ultimately to **device performance (JV curves)**.

The data and methodology are taken and inspired from the paper:  
🔗 [https://www.nature.com/articles/s41524-020-0277-x.pdf](https://www.nature.com/articles/s41524-020-0277-x.pdf)

### Workflow:

1. **Denoising JV Curves (Autoencoder)**:
   - We built an **autoencoder model** to denoise experimental JV curves.
   - The model is trained using **simulated JV curves** with artificially added noise to denoise JV curves.

2. **Regression Model**:
   - Trained on data, **material descriptors vs JV curves** to predict JV characteristics from intermediate descriptors.

3. **Mapping Growth Temperature to Descriptors**:
   - We used an **Arrhenius-like equation** to relate growth temperature to descriptors.
   - The unknown coefficients in this equation were estimated using different methods:
     - **Ensemble Sampler (MCMC using emcee)**
     - **NUTS Sampler (No-U-Turn Sampler using TensorFlow Probability)**
     - **Gradient-based Optimization (using standard gradient descent methods)**

4. **Additional Analysis**:
   - There is also a section that attempts to map both process variables **Growth Temperature and Growth Pressure** to the **material descriptors** using Ensemble Sampler. Another section in the file "The_Total_Work" attempts at using PTSampler which was used by the author of the paper. However that requires an older version of emcee (like v-2.2.1)
      *Note: This section requires additional data on Pressure vs JV curves for execution.*

### Folder Contents:

- `Ensemble_Sampler.ipynb`
- `NUTS_Sampler.ipynb`
- `Regression_Model.ipynb`
- `Denoiserof_JV_Curves.ipynb`
- `Gradient_based_Optimition.ipynb`
- `Growth_Pressure_and_Growth_Temperature`
- `Trained_Models` - Contains regression model and autoencoder.
- `The_Total_Work.ipynb` – A notebook combining all the components.  
      *All models and functions are interdependent and will only work together in this file.*

### Requirements:

- Linux environment
- Python 3.12
- Installed packages:
  - `TensorFlow`
  - `TensorFlow Probability`
  - `emcee`
  - `scikit-learn`
  - `matplotlib`
  - `NumPy`

## PINN_Models

This folder contains Physics-Informed Neural Networks (PINNs) I implemented to solve:

- **1D Laplace Equation**
- **Damped Harmonic Motion**

These models use **gradient descent optimization** to minimize both the physic governing equation's loss and boundary conditions (data) loss for solving the PDEs.

---

