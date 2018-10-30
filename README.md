# Information Bottleneck Algorithms for Relevant-Information-Preserving Signal Processing in Python

## Introduction and Motivation

In its pure form, the information bottleneck method is an unsupervised clustering framework which was first proposed in [TPB99].
Despiting having its origin in machine learning, recently, the information bottleneck method attracted lots of attention in several problems related to communications engineering and digital signal processing.
Applications range from LDPC decoding [LSB16b, LB18, SLB18], receiver design [LSB16], polar code construction [SSB18] to relaying [KK17], C-RAN [CK16], source coding and sensor networks [SLB18].

All these systems are designed to combine coarse quantization and high performance by leveraging the concept of relevant information which is closely related to the information bottleneck method. Applying the information bottleneck method to design signal processing units is termed relevant-information-preserving signal processing [BLS+18].

Nevertheless, we noticed that the lack of public available information bottleneck algorithms discourages people to try to transform their signal processing chain into a relevant-information-preserving one.
Thus, this collection of information bottleneck algorithm is intended to facilitate you, to incorporate the information bottleneck method in your problem.

## Citation

The code is distributed under the MIT license. When using the code for your publication or research please cite this repo as

[SL18] M.Stark, J.Lewandowsky "Information Bottleneck Algorithms in Python", https://goo.gl/QjBTZf


## Installation

1. Download and install Python 3.6 (we recommend Anaconda)
2. Clone the git repository
3. Run python setup.py install to install the ib_base package.
4. Installation requires the following packages:
  * numpy
  * [PyOpenCl.2018](https://documen.tician.de/pyopencl/misc.html)
  * mako
  * [progressbar2](https://pypi.org/project/progressbar2/)
  * cython

## Algorithms and Documentation

This repository contains the following information bottleneck algorithm:
1. Agglomerative information bottleneck algorithm [Slo02]
2. Sequential information bottleneck algorithm [Slo02]
3. Symmetric information bottleneck algorithm (and variants) [LB18]
4. KL-Means information bottleneck algorithm [K17]
5. (planned) Deterministic information bottleneck algorithm [SS17]

A detailed investigation of these algorithm is presented in [HWD17]. A more detailed documentation of all provided functions and a more complete test suite will be available soon.

## Examples
Jupyter notebooks which interactively explains the usage of the algorithms can be found in the folder "Examples".
You can also run the Notebooks in your Browser without any installation!!

### AWGN Channel Quantizer Design
In this example we demonstrate how to use the information bottleneck algorithms to build a channel output quantizer for an AWGN channel
Click here to run the example in your browser: [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fcollaborating.tuhh.de%2Fcip3725%2Fib_base.git/0a1f4f66a1a3c29ff347b2abc79bb292b0be17ca?filepath=Examples%2FAWGNQuantizer.ipynb)

### Author Classification and Text Clustering
One of the first proposed application of the information bottleneck method was document clustering. Based on this idea, in this example we investigate the vocabulary of three famous German authors, i.e., Johann Wolfgang von Goethe, Franz Kafka and Thomas Mann.
Using the information bottleneck method, we demonstrate how to determine word clusters which are very representative for the respective author.
Click here to run the example in your browser: [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fcollaborating.tuhh.de%2Fcip3725%2Fib_base.git/d60712e861972ffb642264ebc1be8c187ffe4ff1?filepath=Examples%2FAuthorClassification.ipynb)

### Binary Clustering of the Titanic Dataset 
(under construction)



## References
[TPB99] N. Tishby, F. C. Pereira, and W. Bialek, “The information bottleneck method,” in 37th annual Allerton Conference on Communication, Control, and Computing, 1999, pp. 368–377.

[Slo02] N. Slonim, “The Information Bottleneck Theory and Applications,” Hebrew University of Jerusalem, 2002.

[LSB16] J. Lewandowsky, M. Stark, and G. Bauch, “Information Bottleneck Graphs for receiver design,” in 2016 IEEE International Symposium on Information Theory (ISIT): IEEE, 2016, pp. 2888–2892.

[LSB16b] J. Lewandowsky, M. Stark, and G. Bauch, “Optimum message mapping LDPC decoders derived from the sum-product algorithm,” in 2016 IEEE International Conference on Communications (ICC).

[BLS+18] G. Bauch, J. Lewandowsky, M. Stark, and P. Oppermann, “Information-Optimum Discrete Signal Processing for Detection and Decoding,” in 2018 IEEE 87th Vehicular Technology Conference (IEEE VTC2018-Spring), Porto, Portugal, 2018.

[SSB18] M. Stark, S. A. A. Shah, and G. Bauch, “Polar Code Construction using the Information Bottleneck Method,” in 2018 IEEE Wireless Communications and Networking Conference Workshops (WCNCW): Polar Coding for Future Networks: Theory and Practice (IEEE WCNCW PCFN 2018), Barcelona, Spain, 2018.

[SLB18] M. Stark, J. Lewandowsky, and G. Bauch, “Iterative Message Alignment for Quantized Message Passing between Distributed Sensor Nodes,” in 2018 IEEE 87th Vehicular Technology Conference (IEEE VTC2018-Spring), Porto, Portugal, 2018.

[LB18] J. Lewandowsky and G. Bauch, “Information-Optimum LDPC Decoders Based on the Information Bottleneck Method,” IEEE Access, vol. 6, pp. 4054–4071, 2018.

[SLB18] M. Stark, J. Lewandowsky, and G. Bauch, “Information-Optimum LDPC Decoders with Message Alignment for Irregular Codes,” in 2018 IEEE Global Communications Conference: Signal Processing for Communications (Globecom2018 SPC), Abu Dhabi, United Arab Emirates, 2018.

[SS17] D. J. Strouse and D. J. Schwab, “The Deterministic Information Bottleneck,” (eng), Neural computation, vol. 29, no. 6, pp. 1611–1630, 2017.

[HWD17] S. Hassanpour, D. Wuebben, and A. Dekorsy, “Overview and Investigation of Algorithms for the Information Bottleneck Method,” in 11th International ITG Conference on Systems, Communications and Coding, 2017.

[KK17] D. Kern and V. Kuehn, “On compress and forward with multiple carriers in the 3-node relay channel exploiting information bottleneck graphs,” in Proceedings 11th International ITG Conference on Systems,
Communications and Coding, Feb 2017, pp. 1–6.

[CK16] D. Chen and V. Kuehn, “Alternating information bottleneck optimization for the compression in the uplink of c-ran,” in 2016 IEEE International Conference on Communications (ICC), May 2016, pp. 1–7.

[K17] B. M. Kurkoski, “On the relationship between the KL means algorithm and the information bottleneck method,” in SCC 2017; 11th International ITG Conference on Systems, Communications and Coding, Feb
2017, pp. 1–6.
