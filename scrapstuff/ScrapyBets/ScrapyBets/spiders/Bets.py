import scrapy


# class BetsSpider(scrapy.Spider):
#     name = "Bets"
#     allowed_domains = ["www.bestfightodds.com"]
#     start_urls = ["https://www.bestfightodds.com"]

#     def parse(self, response):
#         pass

class BetsSpider(scrapy.Spider):
    name = 'Bets'
    allowed_domains = ["www.bestfightodds.com"]
    start_urls = ['https://www.bestfightodds.com']  # Start URL

    def parse(self, response):
        # Extract all hyperlinks
        for href in response.css('a::attr(href)').getall():
            # Construct the absolute URL
            absolute_url = response.urljoin(href)

            # Check if the URL starts with the specified pattern
            if absolute_url.startswith('https://www.bestfightodds.com/events/ufc-'):
                yield {'url': absolute_url}

            # Follow the link if it's within the same domain
            if absolute_url.startswith('https://www.bestfightodds.com'):
                yield scrapy.Request(url=absolute_url, callback=self.parse)
