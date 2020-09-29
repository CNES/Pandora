if [ "x${TRAVIS_REPO_SLUG}" = "xdumasl/Pandora_pandora" ]; then
    openssl aes-256-cbc -K $encrypted_f05cf190aee2_key -iv $encrypted_f05cf190aee2_iv -in .travis/github_deploy_key.enc -out github_deploy_key -d
    chmod 600 github_deploy_key
    eval `ssh-agent -s`
    ssh-add github_deploy_key

    cd "$TRAVIS_BUILD_DIR"
    docker run -v $TRAVIS_BUILD_DIR:/data ldumas/pandora_build_env:ubuntu19.04-python3.7 /bin/sh -c "cd /data; source venv/bin/activate; pip install sphinx sphinx_rtd_theme sphinx_autoapi;"
    docker run -v $TRAVIS_BUILD_DIR:/data ldumas/pandora_build_env:ubuntu19.04-python3.7 /bin/sh -c "cd /data; source venv/bin/activate; cd doc; make html"
    cd $TRAVIS_BUILD_DIR/doc/build/html/
    touch .nojekyll
    git init
    git checkout -b gh-pages
    git add .
    git -c user.name='travis' -c user.email='travis' commit -m init
    git push -q -f git@github.com:${TRAVIS_REPO_SLUG}.git gh-pages &2>/dev/null
    cd "$TRAVIS_BUILD_DIR"
fi