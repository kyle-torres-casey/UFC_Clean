from typing import Any
import scrapy
from scrapy.http import Response


# class UfcfightersSpider(scrapy.Spider):
#     name = "UfcFighters"
#     allowed_domains = ["ufcstats.com"]
#     start_urls = ["http://ufcstats.com/statistics/fighters?char=a&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=b&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=c&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=d&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=e&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=f&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=g&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=h&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=i&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=j&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=k&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=l&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=m&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=n&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=o&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=p&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=q&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=r&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=s&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=t&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=u&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=v&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=w&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=x&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=y&page=all",
#                   "http://ufcstats.com/statistics/fighters?char=z&page=all"]

#     def parse(self, response):
#         for j, row in enumerate(response.css("tr.b-statistics__table-row")):
#             if j > 1:
#                 # Extract and clean text from each column
#                 cols = []
#                 for i, col in enumerate(row.css("td.b-statistics__table-col")):
#                     if i < 3:
#                         # Check if there's an <a> tag and extract its text, otherwise use empty string
#                         link_text = col.css("a.b-link.b-link_style_black::text").get(default='').strip()
#                         cols.append(link_text)
#                     else:
#                         # For columns beyond the third, just extract the text
#                         cols.append(col.css("::text").get(default='').strip())

#                 # Ensure the row has at least the number of columns we expect
#                 cols = cols + [''] * (11 - len(cols))

#                 # Yield the extracted data
#                 item = {
#                     "First": cols[0],
#                     "Last": cols[1],
#                     "Nickname": cols[2],
#                     "Ht": cols[3],
#                     "Wt": cols[4],
#                     "Reach": cols[5],
#                     "Stance": cols[6],
#                     "W": cols[7],
#                     "L": cols[8],
#                     "D": cols[9],
#                     "Belt": cols[10],
#                 }
#                 yield item


# class UfcFightersSpider(scrapy.Spider):
#     name = "UfcFighters"
#     allowed_domains = ["ufcstats.com"]
#     # start_urls = ["http://ufcstats.com/statistics/fighters?char=a&page=all"]
#     start_urls = ["http://ufcstats.com/statistics/fighters?char=a&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=b&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=c&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=d&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=e&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=f&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=g&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=h&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=i&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=j&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=k&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=l&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=m&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=n&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=o&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=p&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=q&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=r&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=s&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=t&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=u&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=v&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=w&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=x&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=y&page=all",
#                 "http://ufcstats.com/statistics/fighters?char=z&page=all"]

#     def parse(self, response):
#         # Extract all fighter detail page links
#         for link in response.css("td.b-statistics__table-col a::attr(href)").getall():
#             full_url = response.urljoin(link)
#             yield scrapy.Request(full_url, callback=self.parse_details)

#     def parse_details(self, response):
#         # Extract statistics from the details page
#         stats = {}
#         stat_elements = response.css("ul.b-list__box-list.b-list__box-list_margin-top li")

#         for stat in stat_elements:
#             title = stat.css("i.b-list__box-item-title::text").get().strip()
#             value = stat.css("::text").re_first(r"\d+\.?\d*%?")
#             if value:
#                 value = value.strip()
#             else:
#                 value = "N/A"  # or some default value if no match is found
#             stats[title] = value

#         yield stats


class UfcFightersSpider(scrapy.Spider):
    name = "UfcFighters"
    allowed_domains = ["ufcstats.com"]
    start_urls = [
        "http://ufcstats.com/statistics/fighters?char=a&page=all",
        "http://ufcstats.com/statistics/fighters?char=b&page=all",
        "http://ufcstats.com/statistics/fighters?char=c&page=all",
        "http://ufcstats.com/statistics/fighters?char=d&page=all",
        "http://ufcstats.com/statistics/fighters?char=e&page=all",
        "http://ufcstats.com/statistics/fighters?char=f&page=all",
        "http://ufcstats.com/statistics/fighters?char=g&page=all",
        "http://ufcstats.com/statistics/fighters?char=h&page=all",
        "http://ufcstats.com/statistics/fighters?char=i&page=all",
        "http://ufcstats.com/statistics/fighters?char=j&page=all",
        "http://ufcstats.com/statistics/fighters?char=k&page=all",
        "http://ufcstats.com/statistics/fighters?char=l&page=all",
        "http://ufcstats.com/statistics/fighters?char=m&page=all",
        "http://ufcstats.com/statistics/fighters?char=n&page=all",
        "http://ufcstats.com/statistics/fighters?char=o&page=all",
        "http://ufcstats.com/statistics/fighters?char=p&page=all",
        "http://ufcstats.com/statistics/fighters?char=q&page=all",
        "http://ufcstats.com/statistics/fighters?char=r&page=all",
        "http://ufcstats.com/statistics/fighters?char=s&page=all",
        "http://ufcstats.com/statistics/fighters?char=t&page=all",
        "http://ufcstats.com/statistics/fighters?char=u&page=all",
        "http://ufcstats.com/statistics/fighters?char=v&page=all",
        "http://ufcstats.com/statistics/fighters?char=w&page=all",
        "http://ufcstats.com/statistics/fighters?char=x&page=all",
        "http://ufcstats.com/statistics/fighters?char=y&page=all",
        "http://ufcstats.com/statistics/fighters?char=z&page=all",
    ]

    def parse(self, response):
        # Extract all fighter detail page links
        for row in response.css("tr.b-statistics__table-row"):
            cols = [col.css("a.b-link.b-link_style_black::text").get(default='').strip() for col in row.css("td.b-statistics__table-col")[:3]]
            cols += [col.css("::text").get(default='').strip() for col in row.css("td.b-statistics__table-col")[3:]]

            if len(cols) >= 11:
                fighter_data = {
                    "First": cols[0],
                    "Last": cols[1],
                    "Nickname": cols[2],
                    "Ht": cols[3],
                    "Wt": cols[4],
                    "Reach": cols[5],
                    "Stance": cols[6],
                    "W": cols[7],
                    "L": cols[8],
                    "D": cols[9],
                    "Belt": cols[10],
                }

                link = row.css("td.b-statistics__table-col a::attr(href)").get()
                if link:
                    full_url = response.urljoin(link)
                    yield scrapy.Request(full_url, callback=self.parse_details, meta={'fighter_data': fighter_data})

    # def parse_details(self, response):
    #     fighter_data = response.meta['fighter_data']
    #     stats = {}
    #     stat_elements = response.css("ul.b-list__box-list.b-list__box-list_margin-top li")

    #     for stat in stat_elements:
    #         title = stat.css("i.b-list__box-item-title::text").get().strip().replace(':', '')
    #         value = stat.css("::text").re_first(r"\d+\.?\d*%?") or "N/A"
    #         stats[title] = value.strip()

    #     fighter_data.update(stats)
    #     yield fighter_data

    def parse_details(self, response):
        fighter_data = response.meta['fighter_data']
        stats = {}
        stat_elements = response.css("ul.b-list__box-list.b-list__box-list_margin-top li")

        for stat in stat_elements:
            title = stat.css("i.b-list__box-item-title::text").get().strip().replace(':', '')
            value = stat.css("::text").re_first(r"\d+\.?\d*%?") or "N/A"
            stats[title] = value.strip()

        # Extract DOB specifically
        dob_label = response.css('i.b-list__box-item-title::text').re_first(r'\s*DOB\s*:')
        if dob_label:
            dob_value = response.xpath(f'//i[contains(text(), "DOB")]/following-sibling::text()').get()
            dob_value = dob_value.strip() if dob_value else 'N/A'
            stats['DOB'] = dob_value

        fighter_data.update(stats)
        yield fighter_data
