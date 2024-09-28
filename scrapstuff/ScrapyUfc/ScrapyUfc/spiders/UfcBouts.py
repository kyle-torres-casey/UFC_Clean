import scrapy

import scrapy

class UfcBoutsSpider(scrapy.Spider):
    name = "UfcBouts"
    allowed_domains = ["ufcstats.com"]
    # start_urls = ["http://ufcstats.com/statistics/events/completed?page=all"]
    start_urls = ["http://ufcstats.com/statistics/events/upcoming?page=all"]
    # http://ufcstats.com/statistics/events/upcoming?page=all
    # start_urls = ["http://ufcstats.com/statistics/events/search?query=ufc+306"]

    def parse(self, response):
        for event in response.css("td.b-statistics__table-col"):
            # print("poop " , event)
            event_name = event.css("i.b-statistics__table-content a::text").get()
            event_link = event.css("i.b-statistics__table-content a::attr(href)").get()
            # print("event_link", event_link)

            yield scrapy.Request(
                response.urljoin(event_link),
                callback=self.parse_event_details,
                meta={"event_name": event_name.strip() if event_name else ""}
            )

    def parse_event_details(self, response):
        event_name = response.meta.get("event_name", "")

        # Extract date and location if available
        date_item = response.css('li.b-list__box-list-item:nth-child(1) i.b-list__box-item-title').xpath('following-sibling::text()').extract_first()
        location_item = response.css('li.b-list__box-list-item:nth-child(2) i.b-list__box-item-title').xpath('following-sibling::text()').extract_first()

        date = date_item.strip() if date_item else ""
        location = location_item.strip() if location_item else ""

        # print("date", date)

        for bout in response.css("tr.b-fight-details__table-row"):
            if bout.css("td.b-fight-details__table-col p.b-fight-details__table-text::text").get():
                # Extract win/loss information for both fighters
                w_l = bout.css('i.b-flag__text').xpath('text()').extract()
                w_l_1 = w_l[0].strip() if len(w_l) > 0 else ""
                w_l_2 = w_l[1].strip() if len(w_l) > 1 else ""

                fighters = bout.css("td.b-fight-details__table-col.l-page_align_left a.b-link::text").getall()

                # print("fighters", fighters)

                # Extract stats for both fighters
                kd = bout.css("td.b-fight-details__table-col:nth-child(3) p.b-fight-details__table-text::text").getall()
                str_stat = bout.css("td.b-fight-details__table-col:nth-child(4) p.b-fight-details__table-text::text").getall()
                td = bout.css("td.b-fight-details__table-col:nth-child(5) p.b-fight-details__table-text::text").getall()
                sub = bout.css("td.b-fight-details__table-col:nth-child(6) p.b-fight-details__table-text::text").getall()
                weight_class = bout.css("td.b-fight-details__table-col:nth-child(7) p.b-fight-details__table-text::text").get(default="").strip()
                method = bout.css("td.b-fight-details__table-col:nth-child(8) p.b-fight-details__table-text::text").get(default="").strip()
                round_num = bout.css("td.b-fight-details__table-col:nth-child(9) p.b-fight-details__table-text::text").get(default="").strip()
                time = bout.css("td.b-fight-details__table-col:nth-child(10) p.b-fight-details__table-text::text").get(default="").strip()

                # Extract the link to the fight details page
                fight_link = bout.css('td.b-fight-details__table-col:nth-child(1) p a::attr(href)').get()

                # print("fight_link", fight_link)

                # Create a dictionary to pass the current data to the next request
                bout_data = {
                    "Event": event_name,
                    "Date": date,
                    "Location": location,
                    "W/L 1": w_l_1,
                    "W/L 2": w_l_2,
                    "Fighter 1": fighters[0].strip() if len(fighters) > 0 else "",
                    "Fighter 2": fighters[1].strip() if len(fighters) > 1 else "",
                    "Kd 1": kd[0].strip() if len(kd) > 0 else "",
                    "Kd 2": kd[1].strip() if len(kd) > 1 else "",
                    "Str 1": str_stat[0].strip() if len(str_stat) > 0 else "",
                    "Str 2": str_stat[1].strip() if len(str_stat) > 1 else "",
                    "Td 1": td[0].strip() if len(td) > 0 else "",
                    "Td 2": td[1].strip() if len(td) > 1 else "",
                    "Sub 1": sub[0].strip() if len(sub) > 0 else "",
                    "Sub 2": sub[1].strip() if len(sub) > 1 else "",
                    "Weight class": weight_class,
                    "Method": method,
                    "Round": round_num,
                    "Time": time,
                }

                if fight_link:
                    # Follow the fight link to get more details
                    yield scrapy.Request(
                        url=response.urljoin(fight_link),
                        callback=self.parse_fight_details,
                        meta={"bout_data": bout_data}
                    )
                else:
                    yield bout_data
                # # Follow the fight link to get more details
                # yield scrapy.Request(
                #     url=response.urljoin(fight_link),
                #     callback=self.parse_fight_details,
                #     meta={"bout_data": bout_data}
                # )

    # def parse_fight_details(self, response):
    #     bout_data = response.meta["bout_data"]

    #     # Extract the fight statistics for each fighter
    #     fighter_1_stats = []
    #     fighter_2_stats = []

    #     for i, stat_col in enumerate(response.css("td.b-fight-details__table-col")):
    #         if i == 0:
    #             continue
    #         stats = stat_col.css("p.b-fight-details__table-text::text").getall()

    #         if len(stats) >= 2:
    #             fighter_1_stats.append(stats[0].strip())
    #             fighter_2_stats.append(stats[1].strip())

    #         # Stop after collecting the first 10 stats for each fighter
    #         if len(fighter_1_stats) == 10 and len(fighter_2_stats) == 10:
    #             break

    #     # Ensure there are exactly 10 stats for each fighter, pad with empty strings if necessary
    #     fighter_1_stats += [""] * (10 - len(fighter_1_stats))
    #     fighter_2_stats += [""] * (10 - len(fighter_2_stats))

    #     # Add these details to the bout_data
    #     bout_data.update({
    #         "Fight Stat 1": fighter_1_stats,
    #         "Fight Stat 2": fighter_2_stats,
    #     })

    #     yield bout_data

    def parse_fight_details(self, response):
        bout_data = response.meta["bout_data"]

        # Extract the fight statistics for each fighter
        fighter_1_stats = []
        fighter_2_stats = []

        column_names = response.css('thead.b-fight-details__table-head tr.b-fight-details__table-row th.b-fight-details__table-col::text').getall()
        column_names = [name.strip() for name in column_names if name.strip()][1:]  # Skip the first empty column name

        for i, stat_col in enumerate(response.css("td.b-fight-details__table-col")):
            if i == 0:
                continue
            stats = stat_col.css("p.b-fight-details__table-text::text").getall()

            if len(stats) >= 2:
                fighter_1_stats.append(stats[0].strip())
                fighter_2_stats.append(stats[1].strip())

            # Stop after collecting the first 9 stats for each fighter
            if len(fighter_1_stats) == 9 and len(fighter_2_stats) == 9:
                break

        # Ensure there are exactly 9 stats for each fighter, pad with empty strings if necessary
        fighter_1_stats += [""] * (9 - len(fighter_1_stats))
        fighter_2_stats += [""] * (9 - len(fighter_2_stats))

        # Add these details to the bout_data, each as a separate field
        for idx in range(9):
            bout_data[column_names[idx] + " 1"] = fighter_1_stats[idx]
            bout_data[column_names[idx] + " 2"] = fighter_2_stats[idx]

        yield bout_data
