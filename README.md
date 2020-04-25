As covid -19 cases are increasing at a very high rate, all over the world.
Even in my country INDIA, where the disease is kept in control by authorities and all the essential service providers.
Despite their best effort we should be ready for worst case scenarios.
--------------------------------------------
This can happen during worst case scenario |
--------------------------------------------
1. Community spread of the virus, to such a extent that number of cases needing hospitaliztion overshoot no. of beds.
2. Large volume of patients will come for testing that it will not be possible to test everyone so rapidly.


Hence models like our, can come handy for screening the person who come for testing.Based on model prediction we can then categorize patients
who will be tested first.

" DISCLAIMER : This model is currently in development and we do not recommend using it for screening purposes as of now.
But if possible do test it and help us grow our current and very limited dataset, so we can make far more better predictions. "


This model predicts covid-19 (if we feed image of X-Ray of patients ), with accuracy of upto 85%.
see deployed model @ www.covidfilter.life

Directory Structure:
1.Train:    |
            |
            |----Covid
            |----NonCovid   (It contains x ray images of pneumonia)
            |----Normal
2.validate: |
            |
            |----Covid
            |----NonCovid   (It contains x ray images of pneumonia)
            |----Normal
3.train.py

4.Test_model.py

>  To train the model : python train.py
>  Model will be saved in .h5 format and Test_model.py can be run to test the model with your own image.

The live deployment of model can be found at : https://www.covidfilter.life

Covid-Net was helpful to me though the architecture of my network is competely different.

Dataset :
The dataset used are:
1.https://github.com/ieee8023/covid-chestxray-dataset
2.https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data

I thank authors of these datasets.

References :
https://arxiv.org/abs/2003.09871 


