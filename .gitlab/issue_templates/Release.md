
# Procédure pour une release Pandora + plugins


Voici la liste des actions pour Pandora:
- [ ] Créer une branche Pandora du nom de la release
- [ ] Tester la branche Pandora
    - [ ] 1. En local
    - [ ] 2. Lancer une CI Jenkins
    - [ ] 3. Le cluster
- [ ] Mettre à jour le changelog de Pandora
- [ ] Merger la branche dans master
- [ ] Créer le tag
- [ ] Gitlab to Github
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
- créer un environnement `virtualenv -q venv`
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

:warning: Pour la suite, vérifier également la version de Python utilisée. Actuellement il s’agit de la 3.8.

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