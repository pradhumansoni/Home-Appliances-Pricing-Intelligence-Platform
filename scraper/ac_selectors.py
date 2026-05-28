"""
Selectors used to parse raw HTML into structured CSV data.
"""

NAME = {
    'tag': 'h2'
}

PRICE = {
    'tag': 'span',
    'attrs': {
        'class': 'price'
    }
}

RATING = {
    'container_tag': 'div',
    'container_attrs': {
        'class': 'rating'
    },

    'value_tag': 'span',
    'value_attrs': {
        'class': 'sm-rating'
    },

    'style_tag': 'style'
}

FEATURES = {
    'container_tag': 'ul',

    'container_attrs': {
        'class': 'sm-feat specs'
    },

    'item_tag': 'li'
}