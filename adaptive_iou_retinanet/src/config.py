class Config:
    def __init__(self):
        ###########################################
        #                 Loging                  #
        ###########################################

        self.log_step_size = 1
        self.img_step_size = 5

        ###########################################
        #                RetinaNet                #
        ###########################################

        # Choose the ResNet architecture
        self.res_architecture = "resnet50"  # resnet50 or resnet101

        # Choose the activation function
        self.relu = "leakyrelu"

        # Choose the normalization layer
        self.norm = "batchnorm"

        # Show an image every n steps during training
        self.show_img_step_size = 1

        # Choose which pyramid levels to extract features from: P1: 1, P2: 2, P3: 3, P4: 4, P5: 5
        self.pyramid_levels = [3, 4]

        # Choose the aspect ratios for the anchor boxes
        self.aspect_ratios = [1 / 2.0, 1 / 1.0, 2 / 1.0]

        # Choose the scale ratios for the anchor boxes
        self.scale_ratios = [1.0, pow(2, 1 / 3.0), pow(2, 2 / 3.0)]

        ###########################################
        #                FocalLoss                #
        ###########################################

        # Choose the alpha and gamma parameters for the Focal Loss function
        self.focalLoss_alpha = 0.25
        self.focalLoss_gamma = 2

        ###########################################
        #           OptimizerSettings             #
        ###########################################

        # Choose the learning rate for the optimizer
        self.learning_rate = 10**-5

        # Choose the weight decay for the optimizer
        self.weight_decay = 10**-8

        ###########################################
        #      Dataloader - DataTransformation    #
        ###########################################

        self.aug = {
            # The size of the images after preprocessing
            "image_size": 400,
            # The initial size of the images before resizing
            "init_resize": 1000,
            # The gamma value for adjusting the brightness of the image
            "gamma_contrast": (0.75, 1.25),
            # The probability of a pixel being dropped out
            "dropout": 0.05,
            # The probability of flipping the image vertically
            "flipud": None,
            # The degree of rotation in increments of 90 degrees
            "rot90": None,  # [1],
            # The percentage of horizontal translation
            "translateX": (-0.05, 0.05),
            # The percentage of vertical translation
            "translateY": (-0.05, 0.05),
            # The range of horizontal scaling
            "scaleX": (0.75, 1.25),
            # The range of vertical scaling
            "scaleY": (0.75, 1.25),
            # The fraction of the image to be cropped
            "crop": 0.05,
            # The range of degrees to rotate the image
            "rotate": (-10, 10),
        }

        # Choose the distribution of the data between train, validation, and test sets
        self.data_distribution = [0.7, 0.2]  # Train, Val, Test (Sum <= 1)

        ################################################################################################################

        # Ensure that the learning rate is less than 1
        assert self.learning_rate < 1

        # Ensure that the weight decay is less than 1
        assert self.weight_decay < 1
