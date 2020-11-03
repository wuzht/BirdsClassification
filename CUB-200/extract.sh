tar -xzf images.tgz
tar -xzf lists.tgz
find . -name ".*" | xargs rm -rf
touch .gitkeep