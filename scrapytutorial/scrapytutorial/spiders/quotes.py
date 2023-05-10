import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"
    allowed_domains = ["quotes.toscrape.com"]
    start_urls = ["http://quotes.toscrape.com/"]

    #1回调方法
    def parse(self, response):
        quotes = response.css('.quote')#获取response
        for quote in quotes:
            item['text'] = quote.css('.text::text').extract_first()#获取第一个text，保存为QuotesItem
            item['author'] = quote.css('.author::text').extract_first()
            item['tags'] = quote.css('.tags .tag::text').extract()#获取全部的tag
            yield item
        
        next = response.css('.pager .next a::attr("herf")').extract_first()#观察网页找到下一个web的css
        url = response.urljoin(next)#获取下一个
        yield scrapy.Request(url=url,callback=self.parse)#一个回调方法，因为下一页结构不变所以也用parse