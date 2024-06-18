# project description
    The project code for the paper "Incorporating adipose tissue into a CT-based deep learning nomogram to differentiate granulomas from lung adenocarcinomas" 

    Includes model training and results display
 
# environmental dependency
    numpy               1.24.3
	pandas              2.0.3
	torch               1.7.0
	torchvision         0.8.0
	scikit-learn        1.3.2
	pyradiomics         3.1.0

# directory structure
    root_path
    Granuloma_Adenocarcinoma_Final
    ├── configs/                            // configuration files
    ├── dataset/                            // the dataset for each cohorts
    │   ├── fat_intrathoracic/
    │   ├── fat_intrathoracic_mask/
    │   ├── intratumor/
    │   ├── merge_region/
    │   ├── peritumor/
    │   └── roi/
    │       ├── img_test0/
    │       ├── img_test1/
    │       ├── img_test2/
    │       └── img_train/
    ├── data                                // initial data
    ├── evaluation_indicators/              // model evaluation index
    ├── evaluation_visualization/           // result visualization
    ├── image_preprocessing_toolkit/        // image preprocessing
    ├── models/                             
    ├── runner/                             // model training
    │   ├── Model_Dict/
    │   ├── Runner_Utils/
    │   ├── Running_Dict/
    │   └── node_cla.py                     // training file
    ├── Utils/                              // project toolkit
    └── README.md                           // help file

# instructions
 	The model training file is located at ./runner/node_cla.py

    Parameters during the model training process are saved in ./runner/Running_Dict

    The best parameters stored in ./runner/Model_Dict

    The evaluation_visualization include important visualization code for the paper
 
 
 