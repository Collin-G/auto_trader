# Portfolio Optimizer 

This project originally took the variance and average of the log returns of multiple stock from the Dow Jones 30 Index and determined their optimal weights in a portfolio. This was done by maximizing their combined Sharpe Ratio while accounting for the covariance between stocks in the portfolio. After being deployed on Azure, the project was able to make bi-weekly buy and hold positions to generate weekly returns of 0.5% on average. This was backtested for 200 decisions made every two weeks from 2015 to 2023.

# Neural Networks and Sentiment Analysis

Currently I am working with a friend from CS to experiment with feeding a series of embeddings of daily news embeddings and stock returns into a neural network to see if I can forecast return and variance. This would allow us to prevent the model from assuming that all events from past will carry over to the future. The model can be seen in the embeds_china_rnd branch. 

Below is the model Architecture.

![image](https://github.com/Collin-G/auto_trader/assets/118686914/dc43303f-ce5e-49f3-99e9-ccae4c4eb814)
