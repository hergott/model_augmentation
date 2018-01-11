import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
import pylab

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense


def mlp(x, y, activation="relu", nodes_per_layer=100, num_layers=5, epochs=1000):
    
    inputs = Input(shape=(1,))
    nn = Dense(nodes_per_layer, activation=activation)(inputs)
      
    for __ in range(num_layers - 1):
        nn = Dense(nodes_per_layer, activation=activation)(nn)  
          
    predictions = Dense(1, activation='linear')(nn)
          
    model = Model(inputs=inputs, outputs=predictions) 
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x, y, batch_size=x.shape[0], epochs=epochs, verbose=0) 

    fitted = model.predict(x, batch_size=x.shape[0], verbose=0)
    
    return model, fitted


def create_data(size=1000):
    
    x = np.linspace(-1.5, 1.5, num=size).reshape(size, 1)
    y = np.sin(x)
    
    distort = (1.0 - np.minimum(1.0, np.abs(x - 0.5))) ** 16.0
    y = y + distort * 0.75
    
    distort2 = (1.0 - np.minimum(1.0, np.abs(x + 0.5))) ** 16.0
    y = y - distort2 * 0.75
    
    data = pd.DataFrame(data=np.hstack([x, y]), columns=["x", "y"])
    ols_fitted = sm.ols(formula="y ~ x", data=data).fit().fittedvalues
    ols_fitted = np.atleast_2d(ols_fitted).T
    
    return x, y, ols_fitted


def format_plot():
    
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    
    plt.xlabel('Observation #')
    plt.ylabel('Y Value')
    plt.grid(color=(1.0, 1.0, 1.0, 0.05), linestyle='--', linewidth=0.2)


def main():
    
    format_plot()

    x, y, ols_fitted = create_data()
    
    y_line, = plt.plot(y, linewidth=6.0, color='#55a86866')

    ols_line, = plt.plot(ols_fitted, linewidth=6.0, color="#c44e5266")
    
    for alpha in np.linspace(0.0, 1.0, 15):
        y_adjusted = y * alpha + ols_fitted * (1.0 - alpha)
        
        model, fitted = mlp(x, y_adjusted)
        
        nn_line, = plt.plot(fitted, linewidth=1.0, color=(0.52, 0.66, 1.0, min(1.0, alpha + 0.05)))   
        
        if alpha == 1.0:
            legnd = plt.legend([y_line, ols_line, nn_line], ['Ground Truth Data', 'Linear Regression Fit', 'Neural Network Fit'], fontsize=25, loc=2)
            for text in legnd.get_texts():
                plt.setp(text, color='w')

    pylab.show()


main()

