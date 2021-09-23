import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error

experience=[[1.0],[1.5],[2.0],[2.5],[3.0],[3.5],[4.0],[4.5],[5.0],[5.5]]
salary=[  10000, 15000 , 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000]

model = Sequential()
model.add(Dense(1000, input_dim=1, activation= "relu"))
model.add(Dense(1000, activation= "relu"))
model.add(Dropout(0.2))
model.add(Dense(500, activation= "relu"))
model.add(Dense(100, activation= "relu"))
model.add(Dense(1))

model.summary()

model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
model.fit(experience,salary,epochs=250)
x=[[8.5]]
print(model.predict(x))
