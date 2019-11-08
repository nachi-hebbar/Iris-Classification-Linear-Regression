import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


iris=datasets.load_iris()
#print(iris)
X=iris.data
y=iris.target
sns.boxplot(x=iris.target ,y=iris.data[:,0])
plt.show()
x_train,x_test,y_train,y_test=train_test_split(X,y)
lin_reg=LinearRegression()
lin_reg=lin_reg.fit(x_train,y_train)
print(lin_reg.score(x_test,y_test))







