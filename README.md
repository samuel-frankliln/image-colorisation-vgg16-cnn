# image-colorisation-vgg16-cnn
ACTUAL IMAGE

![image](https://user-images.githubusercontent.com/44821150/117338337-12f40a00-aebc-11eb-8434-3bdec5b4647a.png)

PREDICTED OUTPUT

![image](https://user-images.githubusercontent.com/44821150/117338393-21422600-aebc-11eb-92d6-329684083fb1.png)


Image colorization is the process of adding color to grayscale images, transforming black-and-white photos into colorful and realistic representations. One approach to image colorization involves using convolutional neural networks (CNNs), specifically the VGG16 architecture, a popular deep learning model initially designed for image classification tasks.

In this process, the VGG16 model is employed to extract meaningful features from grayscale images. The network is pre-trained on large datasets like ImageNet to learn various visual features such as edges, textures, and objects. For colorization, the model is typically modified by removing the fully connected layers and retaining the convolutional layers to serve as a feature extractor. These extracted features are then passed through a series of upsampling or deconvolutional layers to reconstruct the colored image.

The network learns the mapping between the grayscale input and the corresponding colors in the output, producing the ab channels of the LAB color space while the grayscale image serves as the L channel. After training, the model can automatically predict colors for a grayscale image based on its learned features.

This approach leverages the power of deep learning and transfer learning to create high-quality, realistic colorizations, often achieving more detailed and nuanced results compared to traditional hand-crafted methods.
