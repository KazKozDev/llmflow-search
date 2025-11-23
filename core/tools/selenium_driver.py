#!/usr/bin/env python3
"""
LLMFlow Search Agent - Shared Selenium Driver
Provides a centralized way to initialize and manage Selenium WebDriver instances
with consistent configuration for Docker and local environments.
"""

import logging
import os
import random
import time
from typing import Optional

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.common.exceptions import TimeoutException, WebDriverException

# Configure logging
logger = logging.getLogger("selenium_driver")

# Constants
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
]

class SeleniumDriver:
    """
    A shared wrapper for Selenium WebDriver initialization and common operations.
    """
    
    def __init__(self, chromedriver_path: Optional[str] = None, verbose: bool = False):
        """
        Initialize the SeleniumDriver helper.
        
        Args:
            chromedriver_path: Path to the chromedriver executable.
            verbose: Whether to enable verbose logging.
        """
        self.chromedriver_path = chromedriver_path
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    def get_random_user_agent(self) -> str:
        """Return a random User-Agent."""
        return random.choice(USER_AGENTS)

    def get_driver(self) -> Optional[webdriver.Chrome]:
        """Initializes and returns a Selenium WebDriver instance."""
        chrome_options = ChromeOptions()
        
        # --- MANDATORY FLAGS FOR DOCKER ---
        chrome_options.add_argument("--headless")              # Required for Docker
        chrome_options.add_argument("--no-sandbox")            # Required for Linux/Docker
        chrome_options.add_argument("--disable-dev-shm-usage") # Prevent memory crashes
        chrome_options.add_argument("--disable-gpu")           # Accelerate headless mode
        
        # Anti-detection flags
        chrome_options.add_argument(f"--user-agent={self.get_random_user_agent()}")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        # Specify binary location (set in Dockerfile)
        chrome_bin = os.environ.get("CHROME_BIN", "/usr/bin/chromium")
        if os.path.exists(chrome_bin):
             chrome_options.binary_location = chrome_bin

        try:
            # Determine chromedriver path
            chromedriver_path = self.chromedriver_path or os.environ.get("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")
            
            if chromedriver_path and os.path.exists(chromedriver_path):
                service = ChromeService(executable_path=chromedriver_path)
                driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                # Fallback if path not found or not set
                driver = webdriver.Chrome(options=chrome_options)
            
            # Mask webdriver property
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            return driver
        except WebDriverException as e:
            logger.error(f"Failed to initialize Selenium WebDriver: {e}")
            logger.error("Please ensure ChromeDriver is installed and in your PATH, or specify chromedriver_path.")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during WebDriver initialization: {e}")
            return None

    def fetch_page_content(self, url: str, wait_time: tuple = (2, 4)) -> Optional[str]:
        """
        Fetches page content using Selenium.
        
        Args:
            url: The URL to fetch.
            wait_time: Tuple (min, max) seconds to wait after loading.
            
        Returns:
            The HTML content of the page, or None if failed.
        """
        driver = self.get_driver()
        if not driver:
            return None
        
        try:
            logger.debug(f"Fetching URL with Selenium: {url}")
            driver.get(url)
            time.sleep(random.uniform(*wait_time))
            html_content = driver.page_source
            if self.verbose:
                logger.debug(f"Page content fetched with Selenium (length: {len(html_content)})")
            return html_content
        except TimeoutException:
            logger.warning(f"Timeout while loading page with Selenium: {url}")
            return None
        except WebDriverException as e:
            logger.warning(f"WebDriverException while fetching page {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching page {url} with Selenium: {e}")
            return None
        finally:
            if driver:
                driver.quit()
