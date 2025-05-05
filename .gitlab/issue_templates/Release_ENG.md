# Procedure for a Pandora + Plugins Release


Here is the list of actions for Pandora:
- [ ] Create a Pandora branch named after the release (not mandatory as everything can be done on the release branch)
- [ ] Test the branch
    - [ ] 1. Locally
        - Check the Makefile and test the various commands
        - Build the documentation and review it
        - Run all existing tests (especially those not present in the Jenkins CI)
        - Run the notebooks with Jupyter (in addition to the test version) to visually verify the results
        - Run all configuration files present in the data_samples directory and check the result in QGIS
    - [ ] 2. Run a Jenkins CI (be careful with dependencies to use the correct branches) and check all options to run the most comprehensive one.
    - [ ] 3. The cluster
        - Test installation with the Makefile
        - Run the tests
        - Run the configuration files & check the results (QGIS is available in the JupyterHub desktop)
- [ ] Verify that the GitHub CI of the repository is not disabled.
- [ ] Verify that the latest GitHub CI runs on the release branch are not failed. If so, fix them before starting the release.
- [ ] Verify and update the pyproject.toml (dependency versions for Pandora/plugins/etc.)
- [ ] Update the Pandora changelog
- [ ] Update the AUTHORS.md file (if needed)
- [ ] Update the date in the copyright
- [ ] Merge the branch into master (or the new branch into release and then from release into master)
- [ ] Create the tag: :warning: **This must be done from the master branch.**
- [ ] GitLab to GitHub (Nothing to do, just monitor)
- [ ] GitHub CI to verify
- [ ] PyPI to verify
- [ ] Communicate about the release (if needed)

Repeat the same procedure for the plugins:
- [ ] Plugin_libsgm
    - [ ] 1. Libsgm
    - [ ] 2. Plugin_libsgm
- [ ] Plugin_mccnn
    - [ ] 1. MCCNN
    - [ ] 2. Plugin_mccnn
- [ ] Plugin_arnn (not to be done for now)

Below are the explanations for each action for Pandora and the plugins.

:warning: Perform the Pandora release before the plugins.

:exclamation: At the end, there is a section on various errors/issues that may arise.

## Create a Branch
First, create a branch, making sure to start from the **master** branch. Add the necessary commits for the release.

Example command:
```python
git switch -c release-1.2.1 master
```

:warning: Verify the setup.cfg file and, if necessary, update it to ensure compatibility of the release with the appropriate dependency versions.

## Test the Release

The goal here is to verify that the release is functional on multiple platforms. For each of the following subsections, if any errors are encountered, they must be corrected.

### 1. Locally

Start by cloning the repository and on the release branch, verify:

 - that the installation is possible (`make install`)
 - the tests (`make test`)
 - the notebooks (`make test-notebook`) + manual launch to check the displays
 - the file format with `black`, `mypy`, and `pylint` (`make format`, `make lint`)

:warning: These commands may not exist for the plugins. In that case, here are the different commands:

 - create a virtual environment `virtualenv -q venv` or `python3 -m venv venv`
 - source the environment `source venv/bin/activate`
 - perform the installation `pip install -e .[dev,notebook,doc]`
 - run the tests `pytest`
 - create a kernel to test the notebooks `python -m ipykernel install --sys-prefix --name=<test_name> --display-name=<test_name>`
 - test the notebooks `jupyter notebook`
 - run code quality tools with `black` & `lint`

:exclamation: Do not proceed to the next step until verifications are complete.

To proceed to the next step, the release branch must then be pushed to GitLab with the following command:

```shell
`git push origin <branche_a_pousser>`
```

### 2. Run a Jenkins CI

On Jenkins, run a CI build with the Build with parameters option. This ensures that the environment variables overriding the environment are correct.

Make sure to specify the branch created for the release and the required Pandora version.

### 3. The Cluster

This step can be performed in parallel with the previous one.

Here is the list of tasks:

1. Connect to the cluster :
    ```shell
    ssh login@trex.cnes.fr
    ```

2. If the necessary Git repositories for testing are not present, clone them. Here is an example command for the libsgm plugin:
    ```shell
    git clone git@gitlab.cnes.fr:3d/PandoraBox/pandora_plugins/plugin_libsgm.git
    ```

3. Perform the installations

4. Run a Pandora execution with the options, here for the libsgm plugin. Here is an example command to reserve a node before running:
    ```shell
    unset SLURM_JOB_ID
    srun -A cnes_level2 -N 1 -n 8 --time=02:00:00 --mem=64G --x11 --pty bash
    ```
    The scripts for a Pandora execution are in the data_samples folder. Here is an example command without using a plugin:
    ```shell
    pandora a_local_block_matching.json <répertoire_de_sortie>
    ```
    :warning: Note that the path of the images in the JSON file assumes they are in the same directory as the configuration file.

:warning: Also, verify the Python version used. Currently, it is 3.9.

5. Verify that the execution runs normally and check its result.
    - The result directory contains the disparity map(s) and a cfg directory.

6. Run the notebooks.


## Update the Changelog

The CHANGELOG.md file must be modified. To do this:

 - list the tickets fixed for the release,
 - classify them according to the labels #Added, #Fixed, #Changed.

Once the file is updated, add a commit dedicated to this modification.


## Merge the Branch into Master

Before creating the tag, it is important to merge this branch into master.


## Create the tag

Before creating the tag, perform an interactive rebase to clean up the commits if multiple corrections were made previously.

Then, go to GitLab via a browser and navigate to the project in question. Create the tag, making sure to select the master branch.

:boom: A Jenkins CI build is automatically triggered on the project.


## GitLab to GitHub

If an automatically triggered CI build is not green because the dependency on Pandora is incorrect, a new build with the *Build with parameters* option must be performed to avoid manually pushing the branch and tag to GitHub.

To do this, in the *Build with parameters* fields, enter the tag number, for example `refs/tags/<tag_number>`, and specify the correct Pandora version.

At the end of the build, which should normally be successful, the tag is not pushed to GitHub. You must then Replay by modifying the file as follows:

```diff
        stage("update github") {
-        when { expression { return (env.gitlabActionType == 'TAG_PUSH' || env.gitlabSourceBranch == 'master') } }
+       // when { expression { return (env.gitlabActionType == 'TAG_PUSH' || env.gitlabSourceBranch == 'master') } }
...
```

And run a new build. Once completed, go to GitHub to verify that the tag has been pushed. There, the GitHub CI will run another build. If successful, the tag will be pushed to PyPI.

Otherwise, delete the tag on GitHub and then on GitLab. Make the necessary modifications and run a Jenkins CI build on GitLab. Then, resume from the create the **tag section**.


## Known Errors/Issues

Here is a list of known errors/issues:

1. Nothing Happens on GitHub

    Here are the possible causes/sources of this issue:
    - The repository's CI is disabled. Check the date of the last run; if it is more than 24 hours old, there is an issue.
    - The Jenkins CI failed and therefore did not push anything to GitHub.
    - The branch is neither a release branch nor a master branch being tested and therefore will not be present on GitHub (unless forced).
    - The branch (release/master) was rewritten (forced) and therefore does not match the existing history. In this case, delete the GitHub branch and push the new branch on Jenkins. :warning: Only do this in case of emergency.

2. Creating the Tag Causes a Jenkins CI Error

    In this case:
    - Delete the tag that was just created in GitLab.
    - Check Jenkins for the source of the error and correct it (do not do this on master).
    - Update the various branches.
    - Recreate the tag: :warning: **This must be done from the master branch.**

    If an error occurs again, repeat the list of actions above.

3. Creating the Tag Causes a GitHub CI Error

    In this case:
    - Delete the tag that was just created in GitHub & GitLab.
    - Check Jenkins for the source of the error and correct it (do not do this on master). Remember to check the file in the *.github* directory, which is the GitHub CI configuration for the project.
    - Update the various branches.
    - Recreate the tag: :warning: **This must be done from the master branch.**

    If an error occurs again, repeat the list of actions above.
