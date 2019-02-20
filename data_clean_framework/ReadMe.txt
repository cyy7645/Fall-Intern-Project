1. It can handle null value, outliers, duplicated features or records. 

2. There are parameters for users to choose how to handle these cases. 
For string features, I convert it into integer with label encoder or one hot. 

3. For continuous features, I apply chi-square bin method to find its best intervals. 
Like for age, we can divide it into might 6 categories to avoid overfitting. 

4. I select important features according to its information value. 

5. I input the data into models.  

The frame will try every possibility trying to find the best accuracy. 
