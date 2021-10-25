 # MNIST Drilldown -
 ## Aim - 
 1. 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
 2. Less than or equal to 15 Epochs
 3. Less than 10000 Parameters
 4. Achieve this in 3 stages,each with clear targets and results and reasons how to improve the  current result
 ---

## File-1

**Target** : Design a baseline models and simple training/test scripts that has less than 10000 parameters and shows an accuracy of atleast 99.2-99.3% once in 15 epochs. Choose the model/models  to proceed further with.

**Result**
- All of the 6 models designed hit the 99.2% mark, but none of them reached 99.3.
- Net1(7922 parameters) and Net4(9972 parameters) were the best models with highest test accuracy as  - 99.27 and 99.24 respectively and highest train accuracy as 99.47 and 99.5 respectively

**Analysis**
- Net  1 and 4 are the best models because they showed high training accuracy(around 99.4-99.5) in the first 15 epochs.
- Although models 1 and 4 are overfitting, by using regularizations in the next step, I can increase the test accuracy. These 2 models were chosen for further steps because they **learn better** than the other 4.

---
## File-2

**Target** : Use augmentations and regularizations to hit the 99.4% mark atleast once(on the models selected from the previos step only)

**Result**

**Significant reduction in overfitting can be seen in both models**

| model   | highes train accuracy| highest test accuracy |
|---------|----------|---------------|
| model 1 |   99.26  |     99.33     |
| model 4 |   99.2   |     99.33     |

**Analysis**
- Although the none of the models could not hit the 99.4% mark after augmentation the differnce between the training and test accuracy has signifiactly decreasd suggesting the Rotations was a good idea for augmentions
- Still unable to decide which model is better so moving forward with both models
- model enters a plateua or fluctuates around 99.25 in both cases. the loss value does not decrease as well. therfore I have experimentes with LR schedulers in the next step
---
## File-3

**Target** : Use LR scheduler and different optimizers to get a consistent reslut of 99.4% mark

**Result**
 - With different experimentations with staring value of LR, step size and Gamma,**I was able to hit the 99.4 % mark consistly with model 4**
 - Refer notebook for further details.
 - Winning solution
    - Model parameters = 9972
    - starting LR 0.2
    - best train accuracy - 99.4
    - best test accuracy  - 99.44
    - test accuracy in  the last few epochs - (99.37,99.34,99.44,99.42,99.38,99.40)

**Analysis**
 - Much less  overfitting was seen in the final epochs as well (along with consistent 99.4%)
 - Selecting the right LR schema was the key to hit the target
---

