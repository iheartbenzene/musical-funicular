# -*- coding: utf-8 -*-
import scrapy


class RedditbotSpider(scrapy.Spider):
    name = 'redditbot'
    allowed_domains = ['www.reddit.com/r/gameofthrones/']
    start_urls = ['http://www.reddit.com/r/gameofthrones//']

    def parse(self, response):
        titles = response.css('.title.may-blank::text').extract(path='redditbot/')
        votes = response.css('.score.unvoted::text').extract(path='redditbot/')
        times = response.css('time::attr(title)').extract(path='redditbot/')
        comments = response.css('.comments::text').extract(path='redditbot/')
        
        for item in zip(titles, votes, times, comments):
            scraped_information = {
                'title': item[0],
                'vote': item[1],
                'created_at': item[2],
                'comments': item[3]
            }
            yield scraped_information
            