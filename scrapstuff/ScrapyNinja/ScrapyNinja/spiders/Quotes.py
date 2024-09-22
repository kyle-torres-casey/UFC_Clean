import scrapy


class QuotesSpider(scrapy.Spider):
    name = 'Quotes'
    allowed_domains = ['quotes.toscrape.com']
    start_urls = ['http://quotes.toscrape.com/',
                  'http://quotes.toscrape.com/page/2/',
                  'http://quotes.toscrape.com/page/3/',
                  'http://quotes.toscrape.com/page/4/',
                  'http://quotes.toscrape.com/page/5/']

    def parse(self, response):
        for quote in response.css("div.quote"):
            text = quote.css("span.text::text").get()
            author = quote.css("small.author::text").get()
            tags = quote.css("div.tags a.tag::text").getall()

            yield{
                'Text': text,
                'Author': author,
                'Tags': tags
            }
