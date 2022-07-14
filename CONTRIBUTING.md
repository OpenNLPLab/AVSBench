# Contributing

## Development environment set up

Install [bundler](http://bundler.io/):

```
$ gem install bundler
```

Install proejct dependencies using bundler:

```
$ bundle install --path vendor/bundle 
```

Start development server

```
$ bundle exec jekyll serve
```

Open up http://localhost:4000/

(replace website with whatever `baseurl` is set to in `_config.yml`)

## References

* [Jekyll](https://jekyllrb.com/)
* [Jekyll: baseurl explanation](https://byparker.com/blog/2014/clearing-up-confusion-around-baseurl/)
