
# Project Summary -
Bike demand prediction is a common problem faced by bike rental companies, as accurately forecasting the demand for bikes can help optimize inventory and pricing strategies. In this project, I am going to develop a regression sepervised machine learning model to predict the demand for bikes in a given time period.

Orginally dataset of bike rental information form a bike sharing company, had information including details on the number of bikes rented, the time and date of the rental and various weather and seasonality faetures, information on other rented factors that could impact bike demand, such as holidays, functioning and non functioning day.

After processing and cleaning the data, I split it into training and test sets and used the training data to train our machine learning model. I experimented with several different model architectures and hyperparameters settings. Ultimately selecting the model that performed the best on the test data.

To evaluate the performance of our model, I used a variety of metrics, including mean absolute error, root mean squared error and R squared. I found that our model was able to make highly accurate predictions with an R squared value of 0.88 and a mean absolute error of just 2.58.

In addition to evaluating the performance of our model on the test data, I also conducted a series of studies to unsderstand the impact of individual features of the model's performance as well as the weather and seasonality features, had the greatest impact on the bike demand.

Finally, I deployed our model in a live production setting and monitoried its performance over time. I found that the model was able to accurately predict bike demand in real-time, enabling the bike sharing to make informed decisions about inventory and pricing.

# Problem Statement
Currently Rental bikes are introduced in many urban cities for the enhancement of mobility comfort. It is important to make the rental bike available and accessible to the public at the right time as it lessens the waiting time. Eventually, providing the city with a stable supply of rental bikes becomes a major concern. The crucial part is the prediction of bike count required at each hour for the stable supply of rental bikes.

My goal is to develop that is highly accurate, with a low mean absolute error and a high R squared value. The model should be able to provide insights into the factors that most impact bike demand, helping the bike sharing company to make data-driven about how to optimize their operations.


