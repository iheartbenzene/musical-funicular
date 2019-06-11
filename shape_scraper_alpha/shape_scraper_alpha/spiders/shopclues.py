# -*- coding: utf-8 -*-
import scrapy


class ShopcluesSpider(scrapy.Spider):
    name = 'shopclues'
    allowed_domains = ['www.shopclues.com/mobiles-featured-store-4g-smartphone.html']
    start_urls = ['http://www.shopclues.com/mobiles-featured-store-4g-smartphone.html/']

    def parse(self, response):
        pass
