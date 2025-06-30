# Procédure pour une release Pandora + plugins


Voici la liste des actions pour Pandora:
- [ ] Créer une branche Pandora du nom de la release (pas obligatoire car il est possible de tout faire sur la branche release)
- [ ] Tester la branche 
    - [ ] 1. En local 
        - Regarder le Makefile et tester les différentes commandes
        - Builder la documentation et la regarder
        - Lancer tous les tests existants (surtout ceux non présent dans la CI Jenkins)
        - Lancer les notebook avec jupyter (en plus de la version test) afin de vérifier visuellement les résultats
        - Lancer tous les fichiers de configurations présents dans le répertoire data_samples et vérifier sous QGIS le résultat
    - [ ] 2. Lancer une CI Jenkins (attention si dépendance de prendre les bonnes branches) et cocher toutes les options pour en lancer une la plus complète.
    - [ ] 3. Le cluster 
        - Tester de faire une installation avec le Makefile
        - Lancer les tests
        - Lancer les fichiers de configurations & check les résultats (QGIS est disponible dans le desktop de jupyterhub)
- [ ] Vérifier que la CI github du dépôt n'est pas désactivée.
- [ ] Vérifier que les dernières CI github sur release ne sont pas failed. Sinon les corriger avant même de commencer la release.
- [ ] Vérifier et mettre à jour le pyproject.toml (numéro des dépendances Pandora/plugins/etc...)
- [ ] Mettre à jour le changelog de Pandora
- [ ] Mettre à jour le fichier AUTHORS.md (si besoin )
- [ ] Mettre à jour la date dans le copyright
- [ ] Merger la branche dans master (ou bien la nouvelle dans release puis de release dans master)
- [ ] Créer le tag :warning: **Ce dernier doit être fait depuis la branche master.**
- [ ] Gitlab to Github (Rien à faire juste surveiller)
- [ ] CI Github à vérifier
- [ ] Pypi à vérifier
- [ ] Faire une communication sur la release (si besoin)

Refaire la même procédure pour les plugins:
- [ ] Plugin_libsgm
    - [ ] 1. Libsgm
    - [ ] 2. Plugin_libsgm
- [ ] Plugin_mccnn
    - [ ] 1. MCCNN
    - [ ] 2. Plugin_mccnn
- [ ] Plugin_arnn (pas à faire pour le moment)

Ci-dessous les explications pour chacune des actions pour Pandora et les plugins.

:warning: Faire la release de pandora avant celle des plugins.

:exclamation: A la fin il y a une partie sur les différentes erreurs/problèmes que l'on peut avoir.


## Créer une branche
Tout d'abord créer une branche en faisant attention de se mettre sur la branche **master**. Et y ajouter les commits nécessaires ç la release.

Exemple de commande: 
```python
git switch -c release-1.2.1 master
```

:warning: vérifier le fichier setup.cfg et, si nécessaire, le mettre à jour pour assurer la compatibilité de la release avec les versions appropriées des dépendances.


## Tester la release

Le but ici est de vérifier sur plusieurs supports que la release est fonctionnelle. Pour chacune des sous parties ci-dessous, si l'une de lève une/des erreur(s), il est nécessaire de les corriger.

### 1. En local

Commencer par cloner le répertoire et sur la branche de la release vérifier :
 - que l'installation est réalisable (`make install`)
 - les tests (`make test`) 
 - les notebooks (`make test-notebook`) + lancement manuel pour vérifier les affichages
 - le format des fichiers avec `black`, `mypy` et `pylint` (`make format`, `make lint`)

:warning: Il se peut que ces commandes n'existement pas pour les plugins. Dans ce cas voici les différentes commandes:
- créer un environnement `virtualenv -q venv` ou `python3 -m venv venv`
- sourcer l'environnement `source venv/bin/activate`
- réaliser l'installation `pip install -e .[dev,notebook,doc]`
- lancer les tests `pytest`
- créer un noyau pour tester les notebooks `python -m ipykernel install --sys-prefix --name=<nom_test> --display-name=<nom_test>`
- tester les notebooks `jupyter notebook`
- lancer les logiciels de qualité de code avec `black` & `lint`

:exclamation: Ne passer à l'étape suivante qu'une fois les vérifications ok.

Pour passer à l'étape suivante, la branche de release doit alors être poussée sur gitlab, avec la ligne de commande suivante:
```shell
`git push origin <branche_a_pousser>`
```

### 2. Lancer une CI Jenkins

Sur Jenkins, lancer un build de CI avec l'option *Build with parameters*. Cela permet d'être sûr que les variables qui surchargent l'environnement sont les bonnes.

Bien renseigner la branche créée pour la release et la version de pandora nécessaire.

### 3. Le cluster

Cette étape peut-être réalisée en parallèle de la précédente.

Voici la liste des tâches:

1. Se connecter au cluster :
    ```shell
    ssh login@trex.cnes.fr
    ```

2. Si les dépôts git nécessaires au test ne sont pas présents, les cloner. Voici un exemple de commande pour le plugin libsgm:
    ```shell
    git clone git@gitlab.cnes.fr:3d/PandoraBox/pandora_plugins/plugin_libsgm.git
    ```

3. Faire les installations

4. Lancer une execution de pandora avec les options, ici le plugin libsgm. Voici un exemple de commande pour réserver un noeud avant de lancer le run:
    ```shell
    unset SLURM_JOB_ID
    srun -A cnes_level2 -N 1 -n 8 --time=02:00:00 --mem=64G --x11 --pty bash
    ```
    Les scripts pour une execution de Pandora se trouve dans le dossier *data_samples*. Voici un exemple de commande sans utilisation de plugin:
    ```shell
    pandora a_local_block_matching.json <répertoire_de_sortie>
    ```
    :warning: attention le chemin des images dans le json fait que ces dernières sont dans le même dossier que le fichier de configuration.

:warning: Pour la suite, vérifier également la version de Python utilisée. Actuellement il s’agit de la 3.9.

5. Vérifier que l'exécution se déroule normalement ainsi que son résultat.
    - Le dossier résultat contient la(les) carte(s) de disparité(s) ainsi qu'un dossier cfg.

6. Lancer les notebooks.


## Mettre à jour le changelog

Le fichier *CHANGELOG.md* doit être modifié. Pour cela : 
- lister les tickets corrigés pour la release,
- les classer selon le label #Added, #Fixed, #Changed.

Une fois le fichier mis à jour, ajouter un commit dédié à cette modification. 

## Merger la branche dans master

Ici, avant de créer le tag, il est important de merger cette dernière dans master.

## Créer le tag

Avant de créer le tag, faire un rebase interactif pour faire du propre dans les commits si plusieurs corrections ont du être faites précédemments.

Ensuite, aller sur gitlab par le biais d'un browser et se rendre sur le projet en question. Créer le tag en faisant attention de bien choisir la branche master.

:boom: un build de CI Jenkins est automatiquement lancé sur le projet.


## Gitlab to Github

Si un build de CI lancé automatiquement n'est pas vert car la dépendance à pandora n'est pas correcte, un nouveau build avec l'option *Build with parameters* est a réaliser pour ne pas avoir à pousser la branche et le tag à la main sur github.

Pour cela, dans les champs de *Build with parameters* mettre le numéro du tag, par exemple `refs/tags/<numero_du_tag>`, et mettre la bonne version de pandora.

A la fin du build, qui est normalement ok, le tag n'est pas poussé sur github. Il faut alors faire *Replay* en modifiant le fichier comme ci-dessous:

```diff
        stage("update github") {
-        when { expression { return (env.gitlabActionType == 'TAG_PUSH' || env.gitlabSourceBranch == 'master') } }
+       // when { expression { return (env.gitlabActionType == 'TAG_PUSH' || env.gitlabSourceBranch == 'master') } }
...
```
Et relancer un build. Une fois celui-ci terminé, aller sur Github pour vérifier que la tag a bien été poussé. Là, la CI Github va de nouveau effectuer un build. Si celle-ci est ok, alors le tag sera poussé sur pypi.

Dans le cas contraire, supprimer le tag sur Github et ensuite gitlab. Puis, faire les modifications nécessaires et lancer un build de CI Jenkins gitlab. Puis reprendre à partir du chapitre **créer le tag**.


## Erreurs/problèmes connus

Voici une liste des erreurs/problèmes connus:

1. Rien ne se passe sur Github

    Voici une liste des différentes causes/sources à ce problème:
    
    - la CI du dépôt est désactivée. Pour cela, regarder la date du dernier run, si cette dernière date de plus de 24h c'est qu'il y a un hic.
    - la CI Jenkins a failed et n'a donc rien poussé sur Gihub.
    - la branche n'est ni une branche de release ni une branche de master sur laquelle vous testez et donc elle ne sera pas présente sur Github (à par si vous le forcez).
    - la branche (release/master) a été réécrite (en forçant) et donc ne convient pas vace l'historique existant. Il faut alors supprimer la branche Github et pousser sur Jenkins cette nouvelle branche. :warning: A ne faire qu'en cas de force majeur.

2. La création du tag met en erreur la CI Jenkins

    Dans ce cas, il faut :
    
    - supprimer le tag qui vient d'êrte crée dans Gitlab.
    - regarder dans Jenkins la source de l'erreur pour la corriger (ne pas le faire sur master)
    - remettre à jour les différentes branches
    - re-créer le tag :warning: **Ce dernier doit être fait depuis la branche master.**
    
    Si de nouveau une erreur apparaît, re-faire la liste ci-dessus des actions.

3. La création du tag met en erreur la CI github

    Dans ce cas il faut :

    - supprimer le tag qui vient d'êrte crée dans Github & dans Gitlab.
    - regarder dans Jenkins la source de l'erreur pour la corriger (ne pas le faire sur master). Penser à regarder le fichier qui se trouve dans le répertoire *.github* qui est la configuration sur Github de la CI du projet.
    - remettre à jour les différentes branches
    - re-créer le tag :warning: **Ce dernier doit être fait depuis la branche master.**

    Si de nouveau une erreur apparaît, re-faire la liste ci-dessus des actions.
