import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

data = pd.read_csv("SAT_GAT_DATA.csv")
data.head()
data.describe()
data.plot('SAT','GPA',style='o')
plt.ylabel('GPA')
plt.show()

X = data['SAT'].values
Y = data['GPA'].values

st.title("Case Study")
st.write(f"The Problem we have a sample of 84 students, who have studied in college .Their total SAT scores include critical reading, mathematics, and writing. Whereas, the GPA is their Grade Point Average they had at graduation.That’s a very famous relationship. you have to create a linear regression which predicts the GPA of a student based on their SAT score.")

st.write(f"Dataset :- https://drive.google.com/file/d/1U5uvO2PhzchVYp1x7cwUyvqWQl73sDhp/view?usp=sharing")

def Mean(X,Y):
    
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    
    
    return mean_x, mean_y

mean_x, mean_y = Mean(X,Y)


n = len (X)

def SC(n, mean_x, mean_y):

    num = 0
    deno = 0
    for i in range(n):

        num+= (X[i]-mean_x)*(Y[i]-mean_y)
        deno+= (X[i]-mean_x)**2

    m = num/deno

    c = mean_y-(m*mean_x)  
    
    return m, c

m,c = SC(n, mean_x, mean_y)

print("Slope of the equation is:", m)
print("Intercept is:", c)


x = np.linspace(-300,3999,100)
y = m*x+c

def plot(x, y): 

    max_x = np.max(x)
    min_x = np.min(x)

    plt.xlabel('SAT score')
    plt.ylabel('GPA score')
    plt.title('Equation curve')

    plt.plot(x,y,label ='Regression Curve', color = 'blue',linewidth =2)
    plt.scatter(X,Y,label='Scatter Plot',color = 'red')

    plt.legend()
    plt.show()

#plottting graph
st.write(f"Plotting the relationship between SAT and GPA scores")
st.pyplot()

plot(x,y)

#calculate R square value
def rsq(m,c,mean_y):
    a = 0
    b = 0

    for i in range(n):
        y_pred = m*X[i]+c
        a = (Y[i]- mean_y)**2
        b = (y_pred - mean_y)**2

    r2 = (b/a)
    
    return r2

r2 = rsq(m,c,mean_y)
print('Rsquare value is:',r2)

st.write(f'R squared value = {r2}')


from sklearn.model_selection import train_test_split
x = data['SAT'].values.reshape(-1,1)
y = data['GPA'].values.reshape(-1,1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 88)

x_train.shape, y_train.shape, x_test.shape, y_test.shape

from sklearn.linear_model import LinearRegression
reg = LinearRegression()

reg.fit(x_train,y_train)


def predict(x_test):
    y_pred = reg.predict(x_test)
    return y_pred
    
y_pred = predict(x_test)

#comparing actual output values for  x_test with preicted values

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df

#plotting prediction

def plot2(x_test,y_test,y_pred):
    
    plt.scatter(x_test,y_test, label = 'scatter Plot', color='blue', linewidths= 1.5)
    plt.plot(x_test,y_pred,label='Regression Curve', color='black' )
    plt.xlabel("SAT score")
    plt.ylabel('GPA score')
    plt.title('Prediction v/s Test data')
    plt.show()
plot2(x_test,y_test,y_pred)
st.write(f"Prediction v/s Actual Graph")
st.pyplot()


import statsmodels.api as s
x = data['SAT']
y = data["GPA"]
x = s.add_constant(x)
model1 = s.OLS(y,x)
result1 = model1.fit()
print(result1.summary())

st.write(f"OLS Regression Model")

st.write(f"result1.summary()")