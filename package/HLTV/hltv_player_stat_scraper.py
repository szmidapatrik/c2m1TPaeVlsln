from selenium.webdriver.common.by import By
from selenium import webdriver

from time import sleep

import random
import csv




class HLTV_Player_Stat_Scraper:



    def _get_player_profile_links(self, url):
        
        browser = webdriver.Firefox()

        browser.get(url)
        sleep(random.uniform(2, 3))
        allowCookies = browser.find_element(By.ID, 'CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll')
        allowCookies.click()
        sleep(2)

        player_url = []
        
        players = browser.find_elements(By.CSS_SELECTOR, 'table.stats-table tbody tr')
        for player in players:
            link = player.find_element(By.TAG_NAME, 'a').get_attribute('href')
            player_url.append(link)

        browser.quit()
        
        return player_url


    def _get_link_and_decline_cookies(self, url):
        
        browser = webdriver.Firefox()
        
        sleep(0.2)
        browser.get(url)
        sleep(3.2)
        
        allowCookies = browser.find_element(By.ID, 'CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll')
        sleep(random.uniform(0.4, 1.2))
        allowCookies.click()
        
        return browser


    def _get_overview_stats(self, url, stats):
        
        browser = self._get_link_and_decline_cookies(url)
        
        try:
            # Highlighted stats
            stats['player_name'] = browser.find_element(By.CSS_SELECTOR, 'h1.summaryNickname').text
            stats['rating_2.0'] = browser.find_elements(By.CSS_SELECTOR, 'div.summaryStatBreakdownDataValue')[0].text
            stats['DPR'] = browser.find_elements(By.CSS_SELECTOR, 'div.summaryStatBreakdownDataValue')[1].text
            stats['KAST'] = browser.find_elements(By.CSS_SELECTOR, 'div.summaryStatBreakdownDataValue')[2].text[:-1]
            stats['Impact'] = browser.find_elements(By.CSS_SELECTOR, 'div.summaryStatBreakdownDataValue')[3].text
            stats['ADR'] = browser.find_elements(By.CSS_SELECTOR, 'div.summaryStatBreakdownDataValue')[4].text
            stats['KPR'] = browser.find_elements(By.CSS_SELECTOR, 'div.summaryStatBreakdownDataValue')[5].text
            
            # Other stats
            stats['total_kills'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[0].text.replace('Total kills\n', '')
            stats['HS%'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[1].text.replace('Headshot %\n', '')[:-1]
            stats['total_deaths'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[2].text.replace('Total deaths\n', '')
            stats['KD_ratio'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[3].text.replace('K/D Ratio\n', '')
            stats['dmgPR'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[4].text.replace('Damage / Round\n', '')
            stats['grenade_dmgPR'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[5].text.replace('Grenade dmg / Round\n', '')
            stats['maps_played'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[6].text.replace('Maps played\n', '')
            stats['rounds_played'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[7].text.replace('Rounds played\n', '')
            stats['KPR'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[8].text.replace('Kills / round\n', '')
            stats['APR'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[9].text.replace('Assists / round\n', '')
            stats['saved_by_teammatePR'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[11].text.replace('Saved by teammate / round\n', '')
            stats['saved_teammatesPR'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[12].text.replace('Saved teammates / round\n', '')
            
            browser.quit()
            return stats
        except:
            browser.quit()
            raise Exception("error while scraping overview stats")


    def _get_individual_stats(self, url, stats):
        
        url_individual = url.replace('/players/', '/players/individual/')
        browser = self._get_link_and_decline_cookies(url_individual)
        try:
            stats['rounds_with_kils'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[4].text.replace('Rounds with kills\n', '')
            stats['KD_diff'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[5].text.replace('Kill - Death difference\n', '')
            
            stats['total_opening_kills'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[6].text.replace('Total opening kills\n', '')
            stats['total_opening_deaths'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[7].text.replace('Total opening deaths\n', '')
            stats['opening_kill_ratio'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[8].text.replace('Opening kill ratio\n', '')
            stats['opening_kill_rating'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[9].text.replace('Opening kill rating\n', '')
            stats['team_W%_after_opening'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[10].text[:-1].replace('Team win percent after first kill\n', '')
            stats['opening_kill_in_W_rounds'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[11].text[:-1].replace('First kill in won rounds\n', '')
            
            stats['0_kill_rounds'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[12].text.replace('0 kill rounds\n', '')
            stats['1_kill_rounds'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[13].text.replace('1 kill rounds\n', '')
            stats['2_kill_rounds'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[14].text.replace('2 kill rounds\n', '')
            stats['3_kill_rounds'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[15].text.replace('3 kill rounds\n', '')
            stats['4_kill_rounds'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[16].text.replace('4 kill rounds\n', '')
            stats['5_kill_rounds'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[17].text.replace('5 kill rounds\n', '')
            
            stats['rifle_kills'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[18].text.replace('Rifle kills\n', '')
            stats['sniper_kills'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[19].text.replace('Sniper kills\n', '')
            stats['smg_kills'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[20].text.replace('SMG kills\n', '')
            stats['pistol_kills'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[21].text.replace('Pistol kills\n', '')
            stats['grenade_kills'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[22].text.replace('Grenade\n', '')
            stats['other_kills'] = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')[23].text.replace('Other\n', '')
            
            browser.quit()
            return stats
        except:
            browser.quit()
            raise Exception("error while scraping individual stats")


    def _get_match_stats(self, url, stats):
        
        url_match = url.replace('/players/', '/players/matches/')
        browser = self._get_link_and_decline_cookies(url_match)
        try:
            stats['maps_W%'] = browser.find_elements(By.CSS_SELECTOR, 'div.value')[1].text[:-1]
            stats['rating_2.0_1+'] = browser.find_elements(By.CSS_SELECTOR, 'div.value')[2].text[:-1]
            stats['rating_2.0_1+_streak'] = browser.find_elements(By.CSS_SELECTOR, 'div.value')[3].text
                
            browser.quit()
            return stats
        except:
            browser.quit()
            raise Exception("error while scraping match stats")


    def _get_rating_1_0_table(self, url, stats):
        
        url_rating = url.replace('/players/', '/players/career/')
        browser = self._get_link_and_decline_cookies(url_rating)
        try:
            year = browser.find_elements(By.TAG_NAME, 'tr')
            rating_one_stats = {}
            for item in year:
                
                if (item.text.startswith('Past 3 months') or item.text.startswith('Period')):
                    continue
                
                values = item.find_elements(By.TAG_NAME, 'td')
                
                if (len(values) != 5):
                        continue
                
                rating_one_stats['rating_1.0_all_' + values[0].text] = values[1].text
                rating_one_stats['rating_1.0_online_' + values[0].text] = values[2].text
                rating_one_stats['rating_1.0_lan_' + values[0].text] = values[3].text
                rating_one_stats['rating_1.0_major_' + values[0].text] = values[4].text
                
                if item.text.startswith('Career'):
                    stats['rating_1.0_all_' + values[0].text] = values[1].text
                    stats['rating_1.0_online_' + values[0].text] = values[2].text
                    stats['rating_1.0_lan_' + values[0].text] = values[3].text
                    stats['rating_1.0_major_' + values[0].text] = values[4].text
            
            stats['rating_1.0_data'] = rating_one_stats
            browser.quit()
            return stats
        except:
            browser.quit()
            raise Exception("error while rating 1.0 match stats")


    def _get_weapon_stats(self, url, stats):
        
        url_weapon = url.replace('/players/', '/players/weapon/')
        browser = self._get_link_and_decline_cookies(url_weapon)
        try:
            weapon_stats = {}
            weapons = browser.find_elements(By.CSS_SELECTOR, 'div.stats-row')
            for weapon in weapons:
                weapon_name = weapon.text.split('\n')[0].split(' ')[1]
                weapon_stats['kills_' + weapon_name] = weapon.text.split('\n')[1]    
            stats['weapon_data'] = weapon_stats
            
            browser.quit()
            return stats
        except:
            browser.quit()
            raise Exception("error while scraping weapon stats")


    def _get_clutch_stats(self, url, stats):
        
        url_clutch_type_list = ['1on1', '1on2', '1on3', '1on4', '1on5']
        url_clutch = url.replace('/players/', '/players/clutches/')
        url_clutch_array = url_clutch.split('/')
        
        try:
            for clutch_type in url_clutch_type_list:
                
                url_XonX = url_clutch_array.copy()
                url_XonX.insert(7, clutch_type)
                url_clutch = '/'.join(url_XonX)
                browser = self._get_link_and_decline_cookies(url_clutch)
                
                clutches = browser.find_elements(By.CSS_SELECTOR, 'div.value')
                stats['clutches_won_' + clutch_type] = clutches[0].text
                if clutch_type == '1on1':
                    stats['clutches_lost_' + clutch_type] = clutches[1].text 
                
                
                browser.quit()
            
            return stats
        except:
            browser.quit()
            raise Exception("error while scraping clutch stats")


    def _create_file(self, stats, file_path):
        with open(file_path, 'w+', newline='') as f:
            writer = csv.writer(f)
            columns = []
            for x in stats:
                columns.append(x)
            writer.writerow(columns)


    def _write_to_csv(self, stats, file_path):
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            values = []
            for x in stats:
                values.append(stats[x])
            writer.writerow(values)
            

    def _write_url_done(self, url, file_path):
            
        # Save parsed demo to list
        f = open(file_path, 'a+')
        f.write("%s\n" % url)
        f.close()


    def _handle_scrape_error(self, link_with_error, error_txt_path):
        f = open(error_txt_path, 'a+')
        f.write('Error while scraping {}, skipping file\n'.format(link_with_error))
        f.close()
            

    def get_player_stats(self, url, txt_file_path, error_txt_path, csv_file_path, year):

        players = self._get_player_profile_links(url)
        
        for _, url_player in enumerate(players):

            url_player = url_player.split('?')[0] + "?startDate={}-01-01&endDate={}-12-31".format(year, year)

            # Check wether the demo is alreay parsed
            txt_file_path = './{}'.format(txt_file_path)
            with open(txt_file_path) as f:
                lines = f.read().splitlines()
                if url_player in lines:
                    continue
                
            print(url_player)
            
            try:
                stats = self._get_overview_stats(url_player, {})
                stats = self._get_individual_stats(url_player, stats)
                stats = self._get_match_stats(url_player, stats)
                stats = self._get_rating_1_0_table(url_player, stats)
                stats = self._get_weapon_stats(url_player, stats)
                stats = self._get_clutch_stats(url_player, stats)
                
                self._write_to_csv(stats, csv_file_path)
            except:
                self._handle_scrape_error(url_player, error_txt_path)
            self._write_url_done(url_player, txt_file_path)
