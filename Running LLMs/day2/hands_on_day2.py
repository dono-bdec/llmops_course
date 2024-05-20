# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 06:59:07 2024

@author: vishw
"""


#Wikipedia tool
#Install the required libraries
#!pip install --upgrade --quiet  wikipedia

#Import
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

#Initialize the tool
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

#Run the tool
wikipedia.run("Generative Artificial Intelligence")

####################################################################

#YouTube search tool

#Install the required libraries
#%pip install --upgrade --quiet  youtube_search
#%pip install --upgrade --quiet  youtube-transcript-api

#Import
from langchain_community.tools import YouTubeSearchTool
from langchain_community.document_loaders import YoutubeLoader

#Initialize the tool
tool = YouTubeSearchTool()
video = tool.run("LangChain",1) 

#The output of the tool needs regex to obtain the url
import re
urls = re.findall(r'https?://\S+?(?=\')', video)

#Obtain the transcript
loader = YoutubeLoader.from_youtube_url(
    urls[0], add_video_info=False
)

loader.load()