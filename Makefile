.PHONY: check serve

HTML_FILES := $(shell find . -iname '*.html' -type f)

check: $(HTML_FILES) vnu.jar
	java -jar vnu.jar $(HTML_FILES)

serve: vendor
	bundle exec jekyll serve

vendor:
	bundle install --path vendor/bundle

vnu.jar:
	curl -Lf "https://github.com/validator/validator/releases/download/17.9.0/vnu.jar_17.9.0.zip" -o vnu.jar.zip
	unzip -o -d /tmp/vnu vnu.jar.zip
	cp /tmp/vnu/dist/vnu.jar .
