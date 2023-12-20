# Procédure pour le patch avec une version antérieure au dernier tag

Si la version du patch est antérieure à la dernière version réalisée, la procédure sur la création du tag est différente.

Ici, nous prendrons comme exemple pour illustrer la démarche un patch avec pour tag **1.2.1** dans un environnement ayant les tags suivants:

```
Tag : 
     |- 1.4.0
     |- 1.3.0
     |- 1.2.0
     |- 1.1.0
     |- ...
     |- 0.1.0
```

Voici la liste des actions:
- [ ] Créer une branche
- [ ] Tester le patch
    - [ ] 1. En local
    - [ ] 2. Lancer un build sur Jenkins
    - [ ] 3. Le cluster
- [ ] Pousser la branche sur Github
- [ ] Créer le tag
    - [ ] Mettre à jour le changelog
- [ ] Gitlab to Github
- [ ] CI Github à vérifier
- [ ] Pypi à vérifier
- [ ] Faire une communication sur la release (si besoin)

Ci-dessous les explications pour chacune des actions.


## Créer une branche

Tout d'abord créer une branche en faisant attention de se mettre sur la version sur laquelle nous souhaitons appliquer le patch (ici 1.2.0). Et y ajouter les commits nécessaires au patch.

Exemple de commande: 
```python
git switch -c release-1.2.1 1.2.0
```

:warning: vérifier le fichier setup.cfg et, si nécessaire, le mettre à jour pour assurer la compatibilité du patch avec les versions appropriées des dépendances.

## Tester le patch

Le but ici est de vérifier sur plusieurs supports que le patch est fonctionnel. Pour chacune des sous-parties ci-dessous, si l'une d'elles, lève une/des erreur(s), il est nécessaire de les corriger.

### 1. En local

Commencer par cloner le répertoire et sur la branche du patch vérifier :
 - que l'installation est réalisable (`make install`)
 - les tests (`make test`) 
 - les notebooks (`make test-notebook`) + lancement manuel pour vérifier les affichages
 - le format des fichiers avec `black`, `mypy` et `pylint` (`make format`, `make lint`)

:exclamation: Ne passer à l'étape suivante qu'une fois les vérifications ok.

Pour passer à l'étape suivante, la branche du patch doit alors être poussée sur gitlab. Voici la ligne de commande pour l'exemple
```shell
git push origin release-1.2.1
```

### 2. Lancer une CI Jenkins

Sur Jenkins, lancer un build de CI avec l'option *Build with parameters*. Cela permet d'être sûr que les variables qui surchargent l'environnement sont les bonnes.

Bien renseigner la branche créée pour le patch et la version de pandora nécessaire.

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

:sunny: **Si tous les tests local/Jenkins/cluster sont ok, alors il est possible de passer à l'étape suivante**


## Mettre à jour le changelog

Le fichier *CHANGELOG.md* doit être modifié. Pour cela : 
- lister les tickets corrigés pour la release,
- les classer selon le label #Added, #Fixed, #Changed.

Une fois le fichier mis à jour, ajouter un commit dédié à cette modification. 


## Pousser la branche sur Github

Vu qu'une branche a été créée pour le patch, il faut que celle-ci soit sur Github pour que le tag puisse exister ensuite.

Pour cela, il est nécessaire qu'un build dans la CI soit complètement vert pour ne pas avoir à le faire à la main. Un build avec l'option *Build with parameters* est à réaliser. Une fois que celui-ci est entièrement fini et vert, aller sur le numéro de build et cliquer sur le bouton *Replay* dans le menu à gauche.

Il faut alors éditer le Jenkinsfile comme ci-dessous:


```diff
stage("update github") {
-    when { expression { return (env.gitlabActionType == 'TAG_PUSH' || env.gitlabSourceBranch == 'master') } }
+    // when { expression { return (env.gitlabActionType == 'TAG_PUSH' || env.gitlabSourceBranch == 'master') } }
        // push to github
        steps{
            sh """
            // push to our branch from ...
-           git -c http.proxy=http... HEAD:${XXX}
+           git -c http.proxy=http... HEAD:refs/heads/${XXX}
...
```

:warning: La variable dans `{XXX}` est à garder, si elle n'est pas dans l'exemple cela est du à un problème markdown non autorisé.

Et relancer le build sur la CI, normalement une fois le build Jenkins terminé la branche doit se trouver sur Github.

:warning: Une fois sur Github, il faut vérifier que la CI n'est pas désactivée. Pour cela, aller dans *action*, si aucun message d'alerte n'est affiché et que le dernier workflow est récent, c'est que celle-ci est opérationnelle. Sinon faire en sorte de la remettre en route.

## Créer le tag

Avant de créer le tag, faire un rebase interactif pour faire du propre dans les commits si plusieurs corrections ont dû être faites précédemment.

Ensuite, aller sur gitlab par le biais d'un browser et se rendre sur le projet en question. Créer le tag en faisant attention de bien choisir la branche créée à cet effet et non master.

:boom: un build de CI Jenkins est automatiquement lancé sur le projet.


## Gitlab to Github

Si un build de CI lancé automatiquement n'est pas vert car la dépendance à pandora n'est pas correcte, un nouveau build avec l'option *Build with parameters* est à réaliser pour ne pas avoir à pousser la branche et le tag à la main sur github.

Pour cela, dans les champs de *Build with parameters* mettre le numéro du tag, ici dans l'exemple `refs/tags/1.2.1` et mettre la bonne version de pandora.

A la fin du build, qui est normalement ok, le tag n'est pas poussé sur github. Il faut alors faire *Replay* en modifiant le fichier comme ci-dessous:

```diff
        stage("update github") {
-        when { expression { return (env.gitlabActionType == 'TAG_PUSH' || env.gitlabSourceBranch == 'master') } }
+       // when { expression { return (env.gitlabActionType == 'TAG_PUSH' || env.gitlabSourceBranch == 'master') } }
...
```
Et relancer un build. Une fois celui-ci terminé, aller sur Github pour vérifier que le tag a bien été poussé. Là, la CI Github va de nouveau effectuer un build. Si celle-ci est ok, alors le tag sera poussé sur pypi.

Dans le cas contraire, supprimer le tag sur Github et ensuite gitlab. Puis, faire les modifications nécessaires et lancer un build de CI Jenkins gitlab. Puis reprendre à partir du chapitre **créer le tag**.
