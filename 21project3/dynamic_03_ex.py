from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys  # enter키 등을 입력하기위해서
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup
import time

url = 'https://huggingface.co/papers/week/2025-W45'
#웹 드라이버를 자동으로 설치하고 최신버전을 유지
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# 사이트 접속
driver.get(url)
# driver.maximize_window() # 전체 화면으로 실행  옵션
print('사이트 접속했습니다.')
# 사이트가 로드될때까지 기다린다.
time.sleep(1)

# 제목 클릭
more = driver.find_element(By.XPATH,'/html/body/div[1]/main/div[2]/section/div[2]/article[1]/div[2]/div/div[2]/h3/a')
more.click()
time.sleep(2)


# 셀리니움 문법을 이용해서 원하는 태그의 속한 텍스트를 추출
soup = BeautifulSoup(driver.page_source,'html.parser')  
tr_lists = soup.select('#autodanawa_gridC > div.gridMain > article > main > div > table > tbody > tr')
for tr in tr_lists:
    try:
        td_lists = tr.select('td')
        print(td_lists[3].select_one('td.title a').text.strip(), end='\t')
        print(td_lists[4].text.strip(), end='\t')
        print(td_lists[5].text.strip())
        
    except Exception as e:
        pass


time.sleep(10)
# 브라우져 종료
driver.quit()