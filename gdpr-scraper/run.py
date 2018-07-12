import subprocess

subprocess.run(['scrapy', 'runspider', 'scraper.py', '-o', 'policies.json'])


