# Soft Dropout in Quantum Convolutional Neural Network

### 
Quantum convolutional neural network (QCNN), an early application for quantum computers in
the NISQ era, has been consistently proven successful as a machine learning (ML) algorithm for
several tasks with significant accuracy. Derived from its classical counterpart, QCNN is prone to
overfitting. Overfitting is a typical shortcoming of ML models that are trained too closely to the
availed training dataset and perform relatively poorly on unseen datasets for a similar problem.
In this work we study the adaptation of one of the most successful overfitting mitigation method,
knows as the (post-training) dropout method, to the quantum setting. We find that a straightforward
implementation of this method in the quantum setting leads to a significant and undesirable
consequence: a substantial decrease in success probability of the QCNN. We argue that this effect
exposes the crucial role of entanglement in QCNNs and the vulnerability of QCNNs to entanglement
loss. To handle overfitting, we proposed a softer version of the dropout method. We find that the
proposed method allows us to handle successfully overfitting in the test cases.
