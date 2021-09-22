# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 13:54:43 2021

@author: mn12vf
"""

#%%
from selenium import webdriver
#%%
PATH = "C:/Users/mn12vf/chromedriver.exe"
driver = webdriver.Chrome(PATH)
driver.get("https://www.amazon.com/Seagate-Portable-External-Hard-Drive/dp/B07CRG94G3/ref=sr_1_4?dchild=1&qid=1631704510&s=electronics&sr=1-4")

price_id = "priceblock_ourprice"

product_price = driver.find_element_by_id(price_id).text

print(product_price)

#%%
PATH = "C:/Users/mn12vf/chromedriver.exe"
driver = webdriver.Chrome(PATH)
driver.get("https://www.amazon.com/Seagate-Portable-External-Hard-Drive/dp/B07CRG94G3/ref=sr_1_4?dchild=1&qid=1631704510&s=electronics&sr=1-4/checkout/cart")

price_id = "priceblock_ourprice"

product_price_cart = driver.find_element_by_id(price_id).text

print(product_price_cart)

#%%
print("Is the item’s price is the same as the price displayed in the cart?")
print (product_price == product_price_cart)