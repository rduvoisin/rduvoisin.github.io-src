#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals
import os
import sys

AUTHOR = 'Rebeccah Duvoisin'
SITENAME = "Doo /fwä /zAn"
SITEURL = 'https://rduvoisin.github.io' 
# SITEURL = ''

PATHS = 'content'
# STATIC_PATHS = ['images']

# DEFAULT_METADATA = {
#     'status': 'draft',
# }

TIMEZONE = 'America/Chicago'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None
DELETE_OUTPUT_DIRECTORY = True

# Blogroll
LINKS = ( ('View Computation for Public Policy projects on GitHub', 'https://github.com/rduvoisin/Computation_Public_Policy'),
		  ('My GitHub', 'https://github.com/rduvoisin/'))
#          ('Python.org', 'http://python.org/'),
#          ('Jinja2', 'http://jinja.pocoo.org/'),
#          ('You can modify those links in your config file', '#'),)

# Social widget
# SOCIAL = (('You can add links in your config file', '#'),
#           ('Another social link', '#'),)

GITHUB_URL = 'http://github.com/rduvoisin'
DEFAULT_PAGINATION = 4

# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = True


# Plugins 
PLUGIN_PATHS = ['../pelicanplugins/']
PLUGINS = ['pelican-toc',  # This plugin generates a table of contents for pelican articles and pages, 
							# available for themes via article.toc.,
			'extract_toc',
			'code_include' , # Based heavily on docutils.parsers.rst.directives.Include. 
							# Include a file and output as a code block 
							# formatted with pelican's Pygments directive.
			'liquid_tags.include_code',
			'liquid_tags.notebook', 
			# 'ipynb.markup', 
			'ipynb.liquid'
			]
			# 'sitemap', 
			# 'gravatar', 
			# # 'pelican_comment_system', # Pelican Comment System allows you to add static comments to your articles.
			# 'pelican_javascript', 
			# 'pelican_vimeo', 
			# 'pelican_youtube', 
			# 'pelican-cite', # Allows the use of BibTeX citations within a Pelican site.
			# 'pelican-flickr', 
			# 'pelicanfly', # This is pelicanfly, a plugin for Pelican that lets you type things like i ♥ :fa-coffee: in your Markdown documents and have it come out as little Font Awesome icons in the browser. 
			# 'pelican-genealogy', # Traces first and last names associated with the articles.
			# 'pelican-gist',  # Pelican Gist Tag is a library to make it easy to GitHub Gists in your Pelican blogs.
			# 'pelican-githubprojects', # Embed a list of your public GitHub projects in your pages.
			# 'pelican-jinja2content', # This allows the use of Jinja2 template code in articles, 
			# 						 # and thus provides access to for example the extremely 
			# 						 # useful include or import statements of Jinja2 from within articles.
			# 'pelican-langcategory', # Plugin for Pelican to make languages behave the same as categories 
			# 						# (visitor can browse articles in certain language)
			# 'pelican-linkclass', # This plugin allows the setting of the class attribute of <a> elements (generated in Markdown by [ext](link)) according to whether the link is external (i.e. starts with http:// or https://) or internal to the Pelican-generated site.
			# 'pelican-mboxreader', # This pelican plugin adds a generator that can load a Unix style mbox file and generate articles from all the entries in the mailbox.
			# 'pelicanthemes-generator', 
			# 'pelican-version',
			# # 'pelican-open_graph', # This plugin adds Open Graph Protocol tags to your articles.
			# # 'pelican-page-hierarchy', # A Pelican plugin that creates a URL hierarchy for pages that matches the filesystem hierarchy of their sources. 	
			# # 'pelican-page-order',  # A Pelican plugin that adds a page_order attribute to all pages if one is not defined. Allows your templates to sort pages as follows:
			#  ]

TOC = {
    'TOC_HEADERS' : '^h[1-6]',  # What headers should be included in the generated toc
                                # Expected format is a regular expression

    'TOC_RUN'     : 'true'      # Default value for toc generation, if it does not evaluate
                                # to 'true' no toc will be generated
}
# Using pelican-ipynb plugin (pelican_ipynb)
MARKUP = ('md', 'ipynb', 'rst')
# EXTRA_HEADER = open('_nb_header.html').read().decode('utf-8')

# MARKUP = ('md')

# Using liquid tags plugin
NOTEBOOK_DIR = 'notebooks'
# CODE_DIR = 'code'
# EXTRA_HEADER = open('_nb_header.html').read().decode('utf-8')
HEADER_COVER = 'images/bigtahoedark.jpg'

# Themes and additions
# THEME = "../pelicanthemes/elegant"
# THEME = "../pelicanthemes/hyde" # project title offcenter in sidebar
THEME = "../pelicanthemes/clean-blog" # astronaut block
# THEME = "../pelicanthemes/Just-Read" # organized and user-friendly!
# THEME = "../pelicanthemes/medius" # very clean black and white, with right hand sidebar
# THEME = "../pelicanthemes/nest" # blue block. NICE.
# THEME = "../pelicanthemes/pelican-blue" # left blue block meh, no tags
# THEME = "../pelicanthemes/photowall"
# THEME = "../pelicanthemes/twentyhtml5" # Gave an error - 'CRITICAL: UndefinedError: '_' is undefined'


# Footer
# FOOTER_INCLUDE = 'custom_footer.html'
# IGNORE_FILES = [FOOTER_INCLUDE]
# EXTRA_TEMPLATES_PATHS = [os.path.dirname(__file__)]