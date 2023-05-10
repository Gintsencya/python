# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class QuotesItem(scrapy.Item):
    text = scrapy.Field()#定义三个字段
    author = scrapy.Field()
    tags = scrapy.Field()