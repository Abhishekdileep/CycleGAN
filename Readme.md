'''
    Problem trying to be Solved : 
        Image-to-Image Translation : Creating a Dataset that can exchange the existing image to a 
        same scenario of different background can be called Image to Image Translation

        Major Problems : Lack of dataset 
            eg: Lets say you want to translate an image of Summer house to Winter house , 
            you will need a dataset that has been taken photo of on summer and then the same on Winter
            which can be quite complicated given the situation 

        Solution : 
            GAN based model 
                this model has majorly 2 part 
                    A) Generator
                    B) Discriminator
                    The generator is made so that it can fool the Discriminator and at the same time the
                    Discriminator is made such that it can better detect the mistake of the generator 

            GANS 
            domain 1    gen1   Dis1
            domain 2    gen2   Dis2

            Images from gen1 goes to Dis2 and image from gen2 goes to Dis2

            Cycle Consistency 

    Discriminator 
        4 CNN layer 
        Stride value - 2 
        Used Instance Normalization - Takes each channel and sample singularly [imagine weights as a 3-d box , with Height 
                                    , width as one dimension and number of samples as one dimension , channel as another 
                                    then channel and number of samples imagine one singular box top to bottom from there]
    Generator 
        2 Downsampling Layers 
        A few residual (ResNet) Blocks 
        Upsample 2 layer 
        last - Maping to RGB sample 
'''

Dataset to the kaggle link[a link](https://www.kaggle.com/datasets/suyashdamle/cyclegan)