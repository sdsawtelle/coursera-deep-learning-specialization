# coursera-deep-learning-specialization
A collection of course notes and code snippets for the 5-course [Deep Learning specialization](https://www.coursera.org/specializations/deep-learning) on Coursera, developed by deeplearning.ai with the marvelous Andrew Ng. I completed the specialization in July 2019 and my review of the series can be found [here](http://sdsawtelle.github.io/blog/output/coursera-deep-learning-specialization-review.html).

In this repo each of the five courses has a folder containing:
- a jupyter notebook with notes (and images) from the lecture / homework content, where I give a succinct note summary for each section of content. 
- a code-base folder defining functions or code snippets from the homework that I found particularly illuminating, with its own README file annotating the code organization.

The notebook [deep-learning-resources]() is just a place for me to collect links to interesting or useful resources - it is a work in progress.

I did not include the homework notebooks and associated jupyter workspaces in this repo, to respect the Coursera paywall.

## Listing of Course Notebooks
*NOTE: I use custom CSS for image insertion and some other formatting, which is sanitized by the github notebook previewer (see [this article](https://blog.jupyter.org/rendering-notebooks-on-github-f7ac8736d686)). To get a complete preview of my notebooks you can click on the pokeball-looking icon at the top right of the github rendered preview, which will open nbviewer in your browser*

[neural-nets-and-deep-learning](1-course-neural-nets-and-deep-learning/neural-nets-and-deep-learning-course-notebook.ipynb) - An intro to neural networks progressing from logistic regression as an NN, to shallow NNs, to deep L-layer NNs.

[improving-deep-neural-networks](2-course-improving-deep-neural-networks/improving-deep-neural-networks-course-notebook.ipynb) - Various important topics in working with NNs including bias/variance, regularization, optimization methods and backprop considerations and hyperparameter tuning. Also introduced tensorflow + keras.

[structuring-machine-learning-projects](3-course-structuring-machine-learning-projects/structuring-machine-learning-projects-course-notebook.ipynb) - Tips and insights on how to organize and execute a DL project (with many ideas applicable to general ML projects). Covers metrics, train/dev/test splitting, the role of human performance level, how to conduct error analysis, data mismatch between train and dev/test sets, and harnessing transfer learning.

[convolutional-neural-networks](4-course-convolutional-neural-networks/convolutional-neural-networks-course-notebook) - Basic architecture of CNNs including plain-english interpretation and intuition for the role of the different layers / sub-architectures. Survey of some "classic" architectures like AlexNet. Overview of specific architectures like ResNets and Inception and specific use cases like object detection and face recognition. Also covers data augmentation.

[sequence-models](5-course-sequence-models/sequence-models-course-notebook) - Basic architecture of RNNs including plain-english interpretation and intuition for the role of the different layers / sub-architectures. Introduction to, and applications of, embeddings. Overview of specific use cases in sequence-to-sequence architectures.