import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# --- Setup ---
s = Service('C:/Users/Pradhuman/Desktop/chromedriver.exe')
driver = webdriver.Chrome(service=s)
driver.get('https://www.smartprix.com/refrigerators')
time.sleep(3)

# --- Apply Filters ---
driver.find_element(By.XPATH, '//*[@id="app"]/main/aside/div/div[5]/div[2]/label[1]/input').click()
# //*[@id="app"]/main/aside/div/div[5]/div[2]/label[1]/input
time.sleep(1)
driver.find_element(By.XPATH, '//*[@id="app"]/main/aside/div/div[5]/div[2]/label[2]/input').click()
time.sleep(3)

# --- Selectors ---
LOAD_MORE_XPATH = '//*[@id="app"]/main/div[1]/div[2]/div[3]'
CARD_SELECTOR   = 'div[data-way].sm-product'   # ✅ from your HTML inspection

# --- Load More Loop ---
click_count = 0

while True:
    try:
        # Wait until Load More button is present
        load_more_btn = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, LOAD_MORE_XPATH))
        )

        # Scroll it into view
        driver.execute_script(
            "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
            load_more_btn
        )
        time.sleep(1)

        # Count cards BEFORE click
        cards_before = len(driver.find_elements(By.CSS_SELECTOR, CARD_SELECTOR))

        # JS click (bypasses overlay issues)
        driver.execute_script("arguments[0].click();", load_more_btn)
        click_count += 1
        print(f"[Click {click_count}] Cards before: {cards_before}")

        # Wait until new cards actually appear
        WebDriverWait(driver, 15).until(
            lambda d: len(d.find_elements(By.CSS_SELECTOR, CARD_SELECTOR)) > cards_before
        )

        cards_after = len(driver.find_elements(By.CSS_SELECTOR, CARD_SELECTOR))
        print(f"[Click {click_count}] Cards after:  {cards_after}")

    except TimeoutException:
        total = len(driver.find_elements(By.CSS_SELECTOR, CARD_SELECTOR))
        print(f"\nAll data loaded. Total cards: {total}")
        break
    except NoSuchElementException:
        print("Load More button not found. Done.")
        break

# --- Save HTML ---
html = driver.page_source
with open('refrigerator.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("HTML saved to AC.html")
driver.quit()