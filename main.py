import pandas as pd
import numpy as np
import streamlit as st
import requests
from streamlit_lottie import st_lottie
import random
import timeit

import time

from streamlit import image
import imageio
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARIMA
from vector import Vector

import yfinance as yf


from datetime import date


st.set_page_config(page_title="Stock Analysis", page_icon=":bar_chart:",
                   layout="wide", initial_sidebar_state="expanded")
st.subheader("Hi , welcome to ")
st.markdown('<p style="font-size: 50px; color: blue; font-weight: bold; text-align: left;">tradeScope</p>',
            unsafe_allow_html=True)


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_coding = load_lottieurl(
    "https://assets4.lottiefiles.com/packages/lf20_PmGV4skHBv.json")


START = "1800-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
st.title("Trading Watch App")
st_lottie(lottie_coding, speed=1, height=200, key="initial")

stocks = ("AAPL", "GOOG", "MSFT", "GME",
          "AMC", "TSLA", "BTC-USD", "ETH-USD")
picked_stocks = st.selectbox("Select stocks for prediction", stocks)
with st.spinner("Wait for it..."):
    time.sleep(5)
st.success("Done!")
st.balloons()

numberOfDays = st.slider("Number of Days to predict:", 1, 30)
period = numberOfDays * 365


def load_stock_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_loading_mode = st.text("Loading data ***")
data = load_stock_data(picked_stocks)
data_loading_mode.text("Loading data *** Finished Running!")

st.subheader("Live Stock Data 🤑:")
st.write(data.tail())

# prediction
# Load the data into a list and preprocess as needed
# SORTING Data
data2 = data['Close']


# Split the data into a training set and a test set
train_data = data2[:int(len(data2)*0.8)]
test_data = data2[int(len(data2)*0.8):]

# Define a function to calculate the moving average


def moving_average(data2, window_size):
    average = numberOfDays
    for i in range(window_size):
        average += data2[i]
    return average / window_size


# Use the moving average function to make a prediction for tomorrow
tomorrow = moving_average(data2, 3)

# Print the predicted stock price for tomorrow
# Set the title and body of the message

st.write('Predicted stock price increment for',
         numberOfDays, 'days ahead is :', tomorrow)


st.error('DISCLAIMER: This app is for informational purposes only and should not be used for trading. We are not responsible for any liability arising from the use of this app for trading or any other purposes. You are solely responsible for your own trading decisions and any associated risks. By using this app, you acknowledge that you understand and accept these terms.')

st.subheader("Quick Sort Algorithm  with Live Data: ")


# ------------------------------------------------------------


def quicksort(arr):
    # Base case: if the list is empty or has only one element, return it
    if len(arr) <= 1:
        return arr

    # Choose a pivot element
    pivot = arr[0]

    # Divide the list into two sublists:
    # one containing elements that are less than the pivot,
    # and the other containing elements that are greater than or equal to the pivot
    less = [x for x in arr[1:] if x < pivot]
    greater = [x for x in arr[1:] if x >= pivot]

    # Recursively sort the sublists
    less = quicksort(less)
    greater = quicksort(greater)

    # Return the sorted list by combining the sorted sublists and the pivot element
    return less + [pivot] + greater


# Define a list of unsorted elements
arr = data['Close']
# Sort the list using the quicksort algorithm
sorted_arr = quicksort(arr)
last = sorted_arr[-1]
# Print the sorted list
st.write('The highest value of the', picked_stocks, 'stock has been : ', last)
st.write('The minimum value of the', picked_stocks,
         'stock has been : ', sorted_arr[0])

st.subheader("Merge Sort Algorithm with live Stock price: ")
# ------------------------------------------------------------

# MERGE SORT ALGORITHM


def merge_sort(prices):
    # base case: if the list has 1 or 0 elements, it is already sorted
    if len(prices) <= 1:
        return prices

    # split the list into two halves
    mid = len(prices) // 2
    left = prices[:mid]
    right = prices[mid:]

    # recursively sort the two halves
    left = merge_sort(left)
    right = merge_sort(right)

    # merge the sorted halves back together
    return merge(left, right)


def merge(left, right):
    # create an empty list to store the sorted values
    merged = []

    # while both halves have elements remaining,
    # take the smaller of the two elements and append it to the list
    while len(left) > 0 and len(right) > 0:
        if left[0] <= right[0]:
            merged.append(left.pop(0))
        else:
            merged.append(right.pop(0))

    # if one half has remaining elements, append them all to the list
    if len(left) > 0:
        merged.extend(left)
    if len(right) > 0:
        merged.extend(right)

    return merged


# example usage:
prices = data.values.tolist()
sorted_prices = merge_sort(prices)
lastMerged = sorted_prices[-1]


# Print the sorted list
st.write('The highest value of the', picked_stocks,
         'stock has been : ', lastMerged)
st.write('The minimal value of the', picked_stocks,
         'stock has been : ', sorted_prices[0])


# ------------------------------------------------------------
# 100plus values of stocks random generated based on a specific stock
# Print the array of hourly values
stock_price = data.tail(1)['Close'].values[0]

# Generate 100,000 random numbers based on the stock price
numbers = [stock_price * random.random() for _ in range(120_000)]
arrxx = np.array(numbers)


_sortedQuick = quicksort(arrxx)

st.header("Quick Sort Algorithm : ")

avg = np.mean(_sortedQuick)

st.markdown('<p style="font-size: 18px; color: white; font-weight: bold; text-align: left;">Stock random average generator based on stock average price with 120k values using quick sort :</p>',
            unsafe_allow_html=True)
st.write(avg)


st.write("Time elapse: ")
# Print the elapsed time
st.write(timeit.timeit("quicksort(arrxx)",
         setup="from __main__ import quicksort, arrxx", number=1))
quickTimer = timeit.timeit(
    "quicksort(arrxx)", setup="from __main__ import quicksort, arrxx", number=1)


randomMerge = arrxx.tolist()
_sortedMerge = merge_sort(randomMerge)

st.header("Merge Sort Algorithm : ")

avg = np.mean(_sortedMerge)

st.markdown('<p style="font-size: 18px; color: white; font-weight: bold; text-align: left;">Stock random average generator based on stock average price with 120k values using merge sort :</p>',
            unsafe_allow_html=True)
st.write(avg)

st.write("Time elapse: ")
# Print the elapsed time
st.write(timeit.timeit("merge_sort(randomMerge)",
         setup="from __main__ import merge_sort, randomMerge", number=1))
mergeTimer = timeit.timeit("merge_sort(randomMerge)",
                           setup="from __main__ import merge_sort, randomMerge", number=1)

# ------------------------------------------------------------
