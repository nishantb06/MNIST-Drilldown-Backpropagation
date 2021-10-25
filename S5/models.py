import torch
import torch.nn as nn
import torch.nn.functional as F

# Net 1 and Net 4 are the best

#Model -1 7922 parmaeters
dropout_value = 0.02
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=5

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14 rf=6

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(14),
            nn.ReLU()
        ) # output_size = 12 rf=10

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=24, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(dropout_value)
        ) # output_size = 10 rf=14
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=5

        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 5 rf=15

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=15, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(15)
        ) # output_size = 3 rf=18

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        )# output size = 1 rf=34

        self.fc1 = nn.Linear(in_features=15, out_features=10)
        
        

       

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.pool2(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = x.view(-1,15)
        x = self.fc1(x)
        
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


#this model sucks best accuracy 98.89(test) tried to add a GAP at 5th layer with 2 Max pooling layers as well
dropout_value = 0.02
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=5

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14 rf=6

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(14),
            nn.ReLU()
        ) # output_size = 12 rf=10

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(dropout_value)
        ) # output_size = 10 rf=14
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 10 rf=5

        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 7 rf=15

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=15, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(15)
        ) # output_size = 3 rf=18

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1))
        )# output size = 1 rf=34

        self.fc1 = nn.Linear(in_features=15, out_features=10)
        
        

       

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        identity = x
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.pool2(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = x.view(-1,15)
        x = self.fc1(x)
        
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

#A lot less overfitting when we remove Relu from every convolutional layer
#although best accuracy is only 99.13 test
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            #nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            #nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=5

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14 rf=6

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(14),
            #nn.ReLU()
        ) # output_size = 12 rf=10

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=24, kernel_size=(3, 3), padding=0, bias=False),
            #nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(dropout_value)
        ) # output_size = 10 rf=14
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 10 rf=5

        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 5 rf=15

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=15, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(15)
        ) # output_size = 3 rf=18

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        )# output size = 1 rf=34

        self.fc1 = nn.Linear(in_features=15, out_features=10)
        
        

       

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.pool2(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = x.view(-1,15)
        x = self.fc1(x)
        
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


# 9972 params
dropout_value = 0.02
class Net4(nn.Module):
  def __init__(self):
    super(Net4, self).__init__()

    # Input Block  LAYER  1
    self.convblock1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(10),
        nn.Dropout(dropout_value)
        ) # output_size = 28 rf=3

    self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=5
    

    #TRANSITION BLOCK 1
    self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14 rf=6
    self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=5

    #LAYER 2
    self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(14),
            nn.ReLU()
        )
    self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=24, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(dropout_value)
        ) # output_size = 10 rf=14


    #TRANSITION BLOCK 2
    self.pool2 = nn.MaxPool2d(2, 2) # output_size = 5 rf=15
    self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=5
    
    #LAYER 3
    self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=15, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(15)
        ) # output_size = 3 rf=18
    self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=15, out_channels=15, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(15)
        ) # output_size = 3 rf=18

    #GLOBAL AVG POOLINNG
    self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        )# output size = 1 rf=34

    self.fc1 = nn.Linear(in_features=15, out_features=10)

  def forward(self,x):
    x = self.convblock1(x)
    x = self.convblock2(x)

    x = self.pool1(x)
    x = self.convblock3(x)
    
    x = self.convblock4(x)
    x = self.convblock5(x)

    x = self.pool2(x)
    x = self.convblock6(x)
    
    x = self.convblock7(x)
    x = self.convblock8(x)

    x = self.gap(x)
    x = x.view(-1,15)
    x = self.fc1(x)
      
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1)

# 7918 params
dropout_value = 0.02
class Net5(nn.Module):
  def __init__(self):
    super(Net5, self).__init__()

    # Input Block  LAYER  1
    self.convblock1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(10),
        nn.Dropout(dropout_value)
        ) # output_size = 28 rf=3

    self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=5
    

    #TRANSITION BLOCK 1
    self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14 rf=6
    self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=5

    #LAYER 2
    self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(14),
            nn.ReLU()
        )
    self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=18, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Dropout(dropout_value)
        ) # output_size = 10 rf=14


    #TRANSITION BLOCK 2
    self.pool2 = nn.MaxPool2d(2, 2) # output_size = 5 rf=15
    self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=5
    
    #LAYER 3
    self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        ) # output_size = 3 rf=18
    self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # output_size = 3 rf=18

    #GLOBAL AVG POOLINNG
    self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        )# output size = 1 rf=34

    #self.fc1 = nn.Linear(in_features=15, out_features=10)

  def forward(self,x):
    x = self.convblock1(x)
    x = self.convblock2(x)

    x = self.pool1(x)
    x = self.convblock3(x)
    
    x = self.convblock4(x)
    x = self.convblock5(x)

    x = self.pool2(x)
    x = self.convblock6(x)
    
    x = self.convblock7(x)
    x = self.convblock8(x)

    x = self.gap(x)
    # x = x.view(-1,15)
    # x = self.fc1(x)
      
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1)


# 7910 params
dropout_value = 0.02
class Net6(nn.Module):
  def __init__(self):
    super(Net6, self).__init__()

    # Input Block  LAYER  1
    self.convblock1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(10),
        nn.Dropout(dropout_value)
        ) # output_size = 28 rf=3

    self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=5
    

    #TRANSITION BLOCK 1
    self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14 rf=6
    self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=5

    #LAYER 2
    self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(14),
            nn.ReLU()
        )
    self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=18, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Dropout(dropout_value)
        ) # output_size = 10 rf=14


    #TRANSITION BLOCK 2
    self.pool2 = nn.MaxPool2d(2, 2) # output_size = 5 rf=15
    self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=5
    
    #LAYER 3
    self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=15, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(15)
        ) # output_size = 3 rf=18
    # self.convblock8 = nn.Sequential(
    #         nn.Conv2d(in_channels=14, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
    #         nn.ReLU(),
    #         nn.BatchNorm2d(10)
    #     ) # output_size = 3 rf=18

    #GLOBAL AVG POOLINNG
    self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        )# output size = 1 rf=34

    self.fc1 = nn.Linear(in_features=15, out_features=45)
    self.fc2 = nn.Linear(in_features=45,out_features=10)

  def forward(self,x):
    x = self.convblock1(x)
    x = self.convblock2(x)

    x = self.pool1(x)
    x = self.convblock3(x)
    
    x = self.convblock4(x)
    x = self.convblock5(x)

    x = self.pool2(x)
    x = self.convblock6(x)
    
    x = self.convblock7(x)
    #x = self.convblock8(x)

    x = self.gap(x)
    x = x.view(-1,15)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)      
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1)


