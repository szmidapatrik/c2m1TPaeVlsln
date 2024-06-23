from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By 
from selenium import webdriver
from zipfile import ZipFile 
import patoolib

from time import sleep
import random
import os

class HLTV_Demo_Scraper:
    

    
    def get_tournament_demo_links(self, link):
    
        # Allow cookies
        browser = webdriver.Chrome()
        # Remove navigator.webdriver Flag using JavaScript
        browser.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        browser.get(link)
        sleep(random.uniform(3, 5))
        allowCookiesButton = browser.find_element(By.ID, 'CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll')
        sleep(random.uniform(0.5, 0.5))
        allowCookiesButton.click()
        sleep(2)

        # Get the match list items
        matches = browser.find_elements(By.CLASS_NAME, 'result-con')
        matchLinks = []
        for match in matches:
            matchLinks.append(match.find_element(By.CLASS_NAME, 'a-reset').get_attribute('href'))

        # Finish and quit browser, return links
        browser.quit()
        return matchLinks
    

    def get_file_count(self, fileType, folderPath):

        fileCount = 0

        # Iterate directory
        for filename in os.listdir(folderPath):
            if filename.endswith(fileType):
                fileCount += 1
        
        return fileCount
    





    def download_demos(self, tournament, downloadPath):

        with open('done.txt', 'a') as file:
                file.write('-----------------------------------\n')

        # Set download path for Firefox
        firefoxOptions = Options()
        firefoxOptions.set_preference("browser.download.folderList", 2)
        firefoxOptions.set_preference("browser.download.dir", downloadPath)
        firefoxOptions.set_preference("browser.download.useDownloadDir", True)
        firefoxOptions.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/rar,application/zip,application/octet-stream")

        # Iterate through the matches' links
        for matchLink in tournament:

            # Check if the match has already been downloaded
            with open('done.txt', 'r') as file:
                if matchLink in file.read():
                    continue


            # Start browser, allow cookies
            browser = webdriver.Firefox(options=firefoxOptions)
            # Remove navigator.webdriver Flag using JavaScript
            # browser.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            browser.get(matchLink)
            sleep(random.uniform(3, 5))
            allowCookies = browser.find_element(By.ID, 'CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll')
            sleep(random.uniform(0.8, 2.1))
            allowCookies.click()
            sleep(random.uniform(1, 1.9))

            try:

                # Find download button
                downloadLink = browser.find_element(By.CLASS_NAME, 'stream-box').get_attribute('data-demo-link')
                fullDownloadLink = 'https://www.hltv.org' + downloadLink

                fileCount = len(os.listdir(downloadPath))

                # Use JavaScript to initiate the download without blocking
                browser.execute_script("window.location.href = arguments[0];", fullDownloadLink)
                sleep(5)

                while len(os.listdir(downloadPath)) != fileCount + 1 and self.get_file_count('rar.part', downloadPath) != 0:
                    sleep(5)

                # Quit browser
                browser.quit()

            except:
                browser.quit()
                continue

            # Open/Create a txt file and save the done link
            with open('done.txt', 'a') as file:
                file.write(matchLink + '\n')






    def unzip_demo_files(self, sourceFolder, destFolder):

        # Iterate directory
        for zipfile in os.listdir(sourceFolder):

            prefix = zipfile.split('.')[0]
            zipfile = os.path.join(sourceFolder, zipfile)
            if 'sync' in zipfile:
                os.remove(zipfile)
                continue
            patoolib.extract_archive(zipfile, outdir=destFolder)

            # Iterate directory
            for filename in os.listdir(destFolder):
                
                if filename.startswith('_') or filename == 'zip':
                    continue

                os.rename(os.path.join(destFolder, filename), os.path.join(destFolder, '_' + prefix + filename))


    def demo_downloader_for_tournaments(self, link, raw_path, unzipped_path):
        tournament_matches = self.get_tournament_demo_links(link)
        self.download_demos(tournament_matches, raw_path)
        #self.unzip_demo_files(raw_path, unzipped_path)
