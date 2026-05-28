"""
This file is used to save all the selectors used to parse data from raw HTML 
to cleaned csv files.
"""

NAME = {0: "h2"}

PRICE =  {0 : ("span" , {'class':'price'})}


RATING = {0: ('div', {'class': 'rating'}) ,
           1: ('span' ,{'class':'sm-rating'}) ,
             2: 'style'}


FEATURES = {0: ('ul' , {'class': 'sm-feat specs'}),
            1: 'li'}