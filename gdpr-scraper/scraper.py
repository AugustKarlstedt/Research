import scrapy
from scrapy.spiders import Spider, CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.item import Item, Field
from scrapy.http import TextResponse

import re
import time

invalid_characters = re.compile(r'[\\/:"*?<>|\.,]+')

class MySpider(Spider):

    name = 'Privacy Policies'
    #allowed_domains = ['']

    custom_settings = {
        'DOWNLOAD_DELAY': 0.5,
        'DOWNLOAD_TIMEOUT': 30,
        'RETRY_TIMES': 1,
        'AJAXCRAWL_ENABLED': True,
    }

    start_urls = [
        'http://theatlantic.com/privacy-policy/',
        'http://imdb.com/privacy',
        'http://nytimes.com/privacy',
        'http://voxmedia.com/privacy-policy',
        'http://nbcuniversal.com/privacy/full-privacy-policy',
        'http://esquire.com/about/privacy-policy',
        'http://liquor.com/privacy-policy/',
        'http://meredith.com/privacy.html',
        'http://sheknows.com/privacy-policy',
        'http://pbs.org/about/policies/privacy-policy/',
        'http://neworleansonline.com/notmc/privacy.html',
        'http://amazon.com/gp/help/customer/display.html/?nodeId=468496',
        'http://subscription.timeinc.com/storefront/privacy/fortune/generic_privacy_new.html',
        'http://instagram.com/about/legal/privacy/',
        'http://condenast.com/privacy-policy#privacypolicy',
        'http://adweek.com/privacy-policy',
        'http://mlb.mlb.com/mlb/help/mlb_help_about_privacy.jsp',
        'http://disneyprivacycenter.com/privacy-policy-translations/english/',
        'http://washingtonpost.com/privacy-policy/2011/11/18/gIQASIiaiN_story.html',
        'http://foodallergy.org/privacy-policy',
        'http://dictionary.reference.com/privacy',
        'http://legal.kinja.com/privacy-policy-90190742',
        'http://reddit.com/help/privacypolicy',
        'http://subscription.timeinc.com/storefront/privacy/essence/generic_privacy_new.html',
        'http://ocregister.com/privacy/',
        'http://tgifridays.com/privacy',
        'http://corporate.walmart.com/privacy-security/walmart-privacy-policy',
        'http://vikings.com/footer/privacy-policy.html',
        'http://newsbusters.org/privacy-policy',
        'http://washingtonian.com/privacy-policy/',
        'http://barnesandnoble.com/h/help/privacy-policy-complete',
        'http://boardgamegeek.com/privacy',
        'http://fredericknewspost.com/site/privacy.html',
        'http://buffalowildwings.com/en/privacy-policy-terms/',
        'http://kaleidahealth.org/general-information/privacy.asp',
        'http://usa.gov/policies',
        'http://archives.gov/global-pages/privacy.html',
        'http://ifsa-butler.org/about-us/about-ifsa-butler/terms-of-use.html',
        'http://loc.gov/legal/#privacy_policy',
        'http://abita.com/sign_up/privacy_policy',
        'http://coffeereview.com/privacy/',
        'http://communitycoffee.com/termsofuse',
        'http://cariboucoffee.com/footer-folder/privacy',
        'http://google.com/intl/en/policies/privacy/',
        'http://dairyqueen.com/us-en/Privacy-Statement/',
        'http://playstation.com/en-us/legal/privacy-policy/',
        'http://gamestop.com/gs/help/PrivacyPolicy.aspx',
        'http://legalterms.cbsinteractive.com/privacy',
        'http://thefreedictionary.com/privacy-policy.htm',
        'http://randomhouse.com/about/privacy.html',
        'http://restaurantnews.com/privacy-policy/',
        'http://military.com/about-us/privacy-policy',
        'http://tangeroutlet.com/privacy',
        'http://dogbreedinfo.com/privacypolicy.htm',
        'http://minecraft.gamepedia.com/Minecraft_Wiki:Privacy_policy',
        'http://eatchicken.com/privacy-policy.aspx',
        'http://kraftrecipes.com/about/privacynotice.aspx',
        'http://si.edu/privacy',
        'http://education.jlab.org/jhtmllib/JLabSecurityBanner.html',
        'http://lodgemfg.com/page.asp?p_key=0A23CFF9980440FC8D283BC55A36328C',
        'http://ironhorsevineyards.com/Online-Policies/Privacy-Policy',
        'http://sciencemag.org/site/help/about/privacy.xhtml',
        'http://disinfo.com/privacy/',
        'http://ted.com/about/our-organization/our-policies-terms/privacy-policy',
        'http://naturalnews.com/PrivacyPolicy.html',
        'http://everydayhealth.com/privacyterms/#everyday_health_privacy_policy',
        'http://uptodate.com/home/privacy-policy',
        'http://earthkam.ucsd.edu/privacy',
        'http://uh.edu/policies/privacy/',
        'http://research.stlouisfed.org/privacy.html',
        'http://internetbrands.com/privacy/privacy-main.html',
        'http://lynda.com/aboutus/otl-privacy.aspx',
        'http://solarviews.com/eng/privacy.htm',
        'http://mohegansun.com/about-mohegan-sun/privacy-policy.html',
        'http://sci-news.com/privacy-policy.html',
        'http://redorbit.com/privacy_statement/',
        'http://privacy.aol.com/privacy-policy/',
        'http://honda.com/site/site_privacy.aspx',
        'http://privacy.tribune.com',
        'http://highgearmedia.com/privacypolicy',
        'http://static.freep.com/privacy/',
        'http://enthusiastnetwork.com/privacy/',
        'http://allstate.com/about/privacy-statement-aic.aspx',
        'http://acbj.com/privacy/#VI',
        'http://opensecrets.org/about/policy.php',
        'http://dcccd.edu/Pages/PrivacySecurity.aspx',
        'http://gwdocs.com/privacy',
        'http://austincc.edu/web-privacy-statement',
        'http://cincymuseum.org/privacy',
        'http://fool.com/legal/privacy-statement.aspx',
        'http://zacks.com/privacy.php',
        'http://citizen.org/Page.aspx?pid=187',
        'http://bankofamerica.com/privacy/online-privacy-notice.go',
        'http://chasepaymentech.com/privacypolicy',
        'http://thehill.com/privacy-policy',
        'http://policies.yahoo.com/us/en/yahoo/privacy/index.htm',
        'http://miaminewtimes.com/about/privacy-policy',
        'http://rockstargames.com/privacy',
        'http://store.steampowered.com/privacy_agreement/',
        'http://ticketmaster.com/h/privacy.html',
        'http://jibjab.com/about/privacy',
        'http://geocaching.com/about/privacypolicy.aspx',
        'http://taylorswift.com/taylor-privacy-policy/',
        'http://microsoft.com/privacystatement/en-us/BingandMSN/default.aspx',
        'http://post-gazette.com/privacypolicy/',
        'http://sltrib.com/info/privacy/',
        'http://sidearmsports.com/privacypolicy/',
        'http://dailyillini.com/page/privacy',
        'http://wsmv.com/story/18990/this-web-sites-privacy-policy',
        'http://tulsaworld.com/site/privacy.html',
        'http://dailynews.com/privacy',
        'http://lids.com/HelpDesk/Security/PrivacyPolicy',
        'http://sports-reference.com/privacy.shtml',
        'http://foxsports.com/privacy-policy',
        'http://latinpost.com/privacypolicy'
        ]

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse, errback=self.on_error)

    def parse(self, response):
        filename = response.url
        filename = invalid_characters.sub('-', filename)

        with open('html/' + filename + '.html', 'w', encoding='utf-8') as html_file:
            html_file.write(response.text)

        yield {'url': response.url, 'status': 'success'}

    def on_error(self, failure):
        yield {'url': failure.request.url, 'status': 'failure'}
