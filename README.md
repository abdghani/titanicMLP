# titanic
Data Analysis on famous titanic dataset . 


##Description
> - impl1.ipynb  implementation of LogisticRegression, RandomForestClassifier , MLPClassifier ,GaussianNB
> - impl2.py contains implementation of neural networks with accuracy of 0.785 in kaggle

##Implementation
	Z(2) = XW(1)
	a(2) = f(Z(2)) 
	Z(3) = a(2)w(2)
	yhat = f(Z(2))
	J = 1/2 (y-yhat)^2

### layer 2
	dJ/dW(2) = -(y-yhat) * f(Z(3)) * dz(3)/dw(2)
	dJ/dW(1) = d(3) * a(2) <sup>T</sup> 
		
### layer 1
	dJ/dW(1) = d(3) * [W(2)]T  * f'(Z(2)) * X <sup>T</sup> 
	dJ/dW(1) = d(2) * X <sup>T</sup> 

### Demo
[impl2.py](http://thegreyphase.info/#!/home/58bd4a1e9080173cf72b3a2f)

