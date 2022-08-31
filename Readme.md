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
'''



Discriminator 
    4 CNN layer 
        Stride value - 2 
    Idea Name : PatchGAN
    Used Instance Normalization - Takes each channel and sample singularly [imagine weights as a 3-d box , with Height 
                                    , width as one dimension and number of samples as one dimension , channel as another 
                                    then channel and number of samples imagine one singular box top to bottom from there]
    
Model : Discriminator(
  (initial): Sequential(
    (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), padding_mode=reflect)       
    (1): LeakyReLU(negative_slope=0.2)
  )
  (model): Sequential(
    (0): Block(
      (conv): Sequential(
        (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), padding_mode=reflect) 
        (1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)    
        (2): LeakyReLU(negative_slope=0.2)
      )
    )
    (1): Block(
      (conv): Sequential(
        (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), padding_mode=reflect)
        (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)    
        (2): LeakyReLU(negative_slope=0.2)
      )
    )
    (2): Block(
      (conv): Sequential(
        (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), padding_mode=reflect)
        (1): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)    
        (2): LeakyReLU(negative_slope=0.2)
      )
    )
    (3): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), padding_mode=reflect)      
  )
)


Generator 
    2 Downsampling Layers 
    A few residual (ResNet) Blocks 
    Upsample 2 layer 
    last - Maping to RGB sample 



Model Generator(
  (intial): Sequential(
    (0): Conv2d(3, 9, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), padding_mode=reflect)
    (1): ReLU(inplace=True)
  )
  (down_blocks): ModuleList(
    (0): ConvBlock(
      (conv): Sequential(
        (0): Conv2d(9, 18, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), padding_mode=reflect)
        (1): InstanceNorm2d(18, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (2): ReLU(inplace=True)
      )
    )
    (1): ConvBlock(
      (conv): Sequential(
        (0): Conv2d(18, 36, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), padding_mode=reflect)
        (1): InstanceNorm2d(36, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (2): ReLU(inplace=True)
      )
    )
  )
  (residual_blocks): Sequential(
    (0): ResidualBlock(
      (block): Sequential(
        (0): ConvBlock(
          (conv): Sequential(
            (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=reflect)
            (1): InstanceNorm2d(36, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (2): ReLU(inplace=True)
          )
        )
        (1): ConvBlock(
          (conv): Sequential(
            (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=reflect)
            (1): InstanceNorm2d(36, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (2): Identity()
          )
        )
      )
    )
    (1): ResidualBlock(
      (block): Sequential(
        (0): ConvBlock(
          (conv): Sequential(
            (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=reflect)
            (1): InstanceNorm2d(36, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (2): ReLU(inplace=True)
          )
        )
        (1): ConvBlock(
          (conv): Sequential(
            (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=reflect)
            (1): InstanceNorm2d(36, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (2): Identity()
          )
        )
      )
    )
    (2): ResidualBlock(
      (block): Sequential(
        (0): ConvBlock(
          (conv): Sequential(
            (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=reflect)
            (1): InstanceNorm2d(36, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (2): ReLU(inplace=True)
          )
        )
        (1): ConvBlock(
          (conv): Sequential(
            (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=reflect)
            (1): InstanceNorm2d(36, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (2): Identity()
          )
        )
      )
    )
    (3): ResidualBlock(
      (block): Sequential(
        (0): ConvBlock(
          (conv): Sequential(
            (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=reflect)
            (1): InstanceNorm2d(36, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (2): ReLU(inplace=True)
          )
        )
        (1): ConvBlock(
          (conv): Sequential(
            (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=reflect)
            (1): InstanceNorm2d(36, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (2): Identity()
          )
        )
      )
    )
    (4): ResidualBlock(
      (block): Sequential(
        (0): ConvBlock(
          (conv): Sequential(
            (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=reflect)
            (1): InstanceNorm2d(36, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (2): ReLU(inplace=True)
          )
        )
        (1): ConvBlock(
          (conv): Sequential(
            (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=reflect)
            (1): InstanceNorm2d(36, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (2): Identity()
          )
        )
      )
    )
    (5): ResidualBlock(
      (block): Sequential(
        (0): ConvBlock(
          (conv): Sequential(
            (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=reflect)
            (1): InstanceNorm2d(36, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (2): ReLU(inplace=True)
          )
        )
        (1): ConvBlock(
          (conv): Sequential(
            (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=reflect)
            (1): InstanceNorm2d(36, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (2): Identity()
          )
        )
      )
    )
    (6): ResidualBlock(
      (block): Sequential(
        (0): ConvBlock(
          (conv): Sequential(
            (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=reflect)
            (1): InstanceNorm2d(36, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (2): ReLU(inplace=True)
          )
        )
        (1): ConvBlock(
          (conv): Sequential(
            (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=reflect)
            (1): InstanceNorm2d(36, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (2): Identity()
          )
        )
      )
    )
    (7): ResidualBlock(
      (block): Sequential(
        (0): ConvBlock(
          (conv): Sequential(
            (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=reflect)
            (1): InstanceNorm2d(36, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (2): ReLU(inplace=True)
          )
        )
        (1): ConvBlock(
          (conv): Sequential(
            (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=reflect)
            (1): InstanceNorm2d(36, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (2): Identity()
          )
        )
      )
    )
    (8): ResidualBlock(
      (block): Sequential(
        (0): ConvBlock(
          (conv): Sequential(
            (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=reflect)
            (1): InstanceNorm2d(36, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (2): ReLU(inplace=True)
          )
        )
        (1): ConvBlock(
          (conv): Sequential(
            (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=reflect)
            (1): InstanceNorm2d(36, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (2): Identity()
          )
        )
      )
    )
  )
  (up_blocks): ModuleList(
    (0): ConvBlock(
      (conv): Sequential(
        (0): ConvTranspose2d(36, 18, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        (1): InstanceNorm2d(18, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (2): ReLU(inplace=True)
      )
    )
    (1): ConvBlock(
      (conv): Sequential(
        (0): ConvTranspose2d(18, 9, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        (1): InstanceNorm2d(9, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (2): ReLU(inplace=True)
      )
    )
  )
  (last): Conv2d(9, 3, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), padding_mode=reflect)
)