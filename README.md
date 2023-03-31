[![Actions Status](https://github.com/ludgerradke/Adaptive-IoU-RetinaNet/actions/workflows/python3.08.yml/badge.svg)](https://github.com/ludgerradke/Adaptive-IoU-RetinaNet/actions/workflows/python3.08.yml/badge.svg)
[![Actions Status](https://github.com/ludgerradke/Adaptive-IoU-RetinaNet/actions/workflows/python3.09.yml/badge.svg)](https://github.com/ludgerradke/Adaptive-IoU-RetinaNet/actions/workflows/python3.09.yml/badge.svg)

# Adaptive-IoU-RetinaNet

This repository contains a modified version of the source code for the paper "Adaptive IoU Thresholding for Improving Small Object Detection: A Proof-of-Concept Study of Hand Erosions Classification of Patients with Rheumatic Arthritis on X-ray Images" (DOI: [10.3390/diagnostics13010104](https://www.mdpi.com/2075-4418/13/1/104). 
The purpose of this modification is to remove patient filter criteria and provide a test dataset consisting of circles with different sizes and brightness levels for classification testing.

![](/assets/paper.png)

### Test dataset
The test dataset used in this repository consists of images of different shapes, where each shape consists of an outer and an inner shape. The outer shape (score_1) represents the environment (hence as in the paper the joint), while the inner shape (score_2) represents the aquivalent to erosion. In each image, each score_1 always occurs exactly once, whereas score_2 is random.

Score_1 and score_2 are defined on the basis of the shape, whereas score_2 can only take the values 1 to 3:

- 1 = rectangle
- 2 = circle
- 3 = line
- 4 = ellipse
- 5 = triangle
- 6 = pentagon
- 7 = Hexagon

The RetinaNet model in this repository attempts to recognise the shapes based on both scores.
#### Ground truth
![](/assets/gt_image.png)

#### Prediction after 20 Epochs with adaptive adjustment

![](/assets/with_adjustment.png)

#### Prediction after 20 Epochs without adaptive adjustment

![](/assets/without_adjustment.png)

### Installation
To install the required dependencies for this repository, use the following command:

````bash
pip install -r requirements.txt
````
This will install all the necessary packages including PyTorch, Torchvision, Matplotlib, NumPy, Pandas, Scikit-learn, Pydicom, and Pillow. 
The `requirements.txt` file also specifies the version number of each package to ensure compatibility. 
Once the dependencies are installed, you can run the code in this repository.

### Usage

To run the RetinaNet model on the test dataset, simply run the train.py script. 
The script accepts the following command-line arguments:

````bash
python train.py
````

**User Input for train.py**

There are several arguments that can be passed to train.py when running, such as the number of GPUs used or the number of training epochs. A full list of arguments and their default values can be found in the get_args_windows function in the train.py script.

**Modification**
To use this provided RetinaNet with adaptive IoU-values on your own dataset, you only need to create a new dataset class.
Please see the abstract dataset class for the correct implementation and all nessesary functions

### Configuration
The configuration settings of the RetinaNet model, such as the ResNet architecture used or the activation function, can be changed in the Config class in the train.py script. Additional information can be found in the comments in the Config class.

### License
This repository is released under the MIT license. See the 'LICENSE' file for details.

### Citation
If you use this code, please cite our paper: "Adaptive IoU Thresholding for Improving Small Object Detection: A Proof-of-Concept Study of Hand Erosions Classification of Patients with Rheumatic Arthritis on X-ray Images".
### Contact
If you have any questions, feedback or cooperation requests, please don't hesitate to contact me on [GitHub](https://github.com/ludgerradke), [LinkedIn](https://www.linkedin.com/in/ludger-radke) or [ResearchGate](https://www.researchgate.net/profile/Karl-Radke-2).