# FLSwitch

## Intro

A fast aggregation framework for secure Federated Learning (FL) with both security and efficiency, customizing different security protections for different learning phases.

FLSwitch consists of three main components:

- The implementation of the current popular batch homomorphic encryption (HE) aggregation scheme -- BatchCrypt scheme 
- The implementation of fast aggregation protocol
- Two rules based on threshold switching and meta-learning switching

The basic idea of the fast aggregation protocol is to separate the learning parameters into *anchor* and *residue* in the near convergence stage of machine learning, when anchors having  a high common rate. Then part of clients are appointed to transmit the cipher of the common anchors and the plaintext of residues. This protocol ensures the safety and parameter homomorphism and balance the computation load. A full description of FLSwitch can not be presented because the relevant paper is under review. The part covered by the `covered` mark cannot be described in detail at this time.

## Dependency libraries

Python >= 3.7

pytorch >= 1.7.0

Encryption & Decryption：gmpy2==2.0.8, numpy

## Example

### Training

Run `main.py`, clients and server can be directly called to coordinate encryption training network.

1. `CryptNet_client.py` contains local model training and the process of encryption and upload of gradients.
3. `CryptNet_server.py` contains the aggregation of encrypted gradients.

Parameters for training are set in `train_params.py`：

- **BACKEND**: encryption scheme，can be "plain" and "batchcrypt"

  plain means plaintext，batchcrypt means the encryption algorithm based on BatchCrypt scheme

- **DATASET**: training dataset can be MNIST, FASHION-MNIST, CIFAR10 and CIFAR100
- **CLIENT_NUM**: the number of client
- **SWITCH_MODE**: switching rule, can be "thre"(threshold switching) and "pred"(meta-learning switching)
- **CON_ACC_DICT**: threshold value for different datasets under the threshold switching rule
- **SWITCH_MODEL_PATH**：pre-trained meta-learning switching model `covered`
- **PARASCOUNT**: the selection process of anchor `covered`
- **K、SPARSE**：hyperparameters to control the anchor selection  `covered`

Other parameters are preset as default.

### Results records

The `log` folder records the running logs of each client and the server, the `result` folder records the loss and accuracy of each epoch of FL, and the `choice` folder records the client selection and transmission parameter ratio under the fast aggregation protocol.

## Other problems

- If an error occurs：OSError: [Errno 98] Address already in use

  The cause is that the server port is occupied. Please change the **SERVER_PORT** to any free port in `train_params.py`.

