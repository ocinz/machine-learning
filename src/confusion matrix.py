import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/ecommerce_banner_promo.csv')
print(data.head(20))

X = data.drop(['Ad Topic Line','City','Country','Timestamp','Clicked on Ad'], axis = 1)
y = data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

logreg = LogisticRegression(solver='newton-cholesky')
logreg = logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

#apply confusion_matrix function to y_test and y_pred
print("Confusion matrix:")
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
cm_display.plot()
plt.show()