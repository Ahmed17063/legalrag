# Configurer la clé d'API Voyage AI

Guide pas à pas pour donner au pipeline accès à l'API d'embeddings.

**Durée : environ 5 minutes.** Aucune carte bancaire requise pour démarrer —
mais lire la section [Plafonds de débit](#plafonds-de-débit), qui change
complètement le temps d'exécution.

---

## Ce qu'on va faire

```
┌──────────────────┐        ┌──────────────┐        ┌─────────────────────┐
│  Compte          │        │  Clé d'API   │        │  Endpoint           │
│  MongoDB Atlas   │───────▶│  (secrète,   │───────▶│  déduit du préfixe  │
│  ou Voyage       │  crée  │  affichée    │ utilisée│  de la clé          │
└──────────────────┘        │  une fois)   │  par   └─────────────────────┘
                            └──────────────┘
                                   │
                                   ▼
                            .env : VOYAGE_API_KEY
```

Deux idées à retenir :

1. **La clé n'est affichée qu'une seule fois.** Il faut la copier
   immédiatement ; sinon, il faut en générer une autre.
2. **Le domaine à appeler dépend de l'origine de la clé.** Voyage AI
   appartient maintenant à MongoDB, et les deux plateformes coexistent. Se
   tromper de domaine produit une erreur d'authentification dont le message
   n'indique pas la cause.

---

## Étape 1 — Créer un compte

Il existe deux parcours. **Le premier est celui à suivre aujourd'hui**, le
second reste valable si vous avez déjà un compte.

### Parcours A — MongoDB Atlas (actuel)

Aucun cluster payant n'est nécessaire, et il n'est pas nécessaire d'être
client de la base de données MongoDB.

1. Créer un compte sur [cloud.mongodb.com](https://cloud.mongodb.com)
2. Atlas crée automatiquement une organisation et un projet

### Parcours B — Dashboard Voyage (historique)

Les comptes créés sur [dashboard.voyageai.com](https://dashboard.voyageai.com/)
restent pleinement fonctionnels. Leurs clés commencent par `pa-`.

---

## Étape 2 — Générer la clé

### Depuis MongoDB Atlas

1. Dans la barre de navigation, sous **Services**, cliquer sur
   **Model API Keys**
2. **Create new model API key**
3. Donner un nom à la clé (par exemple `legalrag`)
4. Sélectionner le projet
5. **Cloud provider** et **Geography** : laisser sur `any`

   > Restreindre une clé à une zone géographique impose un palier payant, et
   > oblige ensuite à viser un domaine différent. Sans raison particulière,
   > laisser `any`.

6. **Create**
7. **Copier la clé immédiatement**, puis *Done*

### Depuis le dashboard Voyage

*API Keys* → **Create new secret key** → copier.

> ⚠️ **La clé n'est plus jamais réaffichée.** Après avoir quitté la page,
> il n'y a aucun moyen de la relire. En cas de perte, en créer une nouvelle
> et supprimer l'ancienne.

---

## Étape 3 — Comprendre quel domaine sera appelé

C'est le point qui coûte le plus de temps quand on l'ignore.

| Origine de la clé | Préfixe | Domaine à appeler |
|---|---|---|
| Atlas, non restreinte | *aucun* | `ai.mongodb.com` |
| Atlas, Europe | `al-eu-` | `eu.ai.mongodb.com` |
| Atlas, États-Unis | `al-us-` | `us.ai.mongodb.com` |
| Ancienne plateforme Voyage | `pa-` | `api.voyageai.com` |

**Vous n'avez normalement rien à faire** : le pipeline déduit le domaine du
préfixe de la clé, dans
[`resolve_base_url`](../src/legalrag/config.py).

La variable `VOYAGE_API_BASE_URL` permet de forcer une URL si jamais ce
choix automatique se trompe — la laisser vide dans le cas général.

---

## Étape 4 — Renseigner la configuration

Dans le fichier `.env` à la racine du dépôt, créé depuis `.env.example` :

```
VOYAGE_API_KEY=<la clé copiée>
VOYAGE_API_BASE_URL=
```

| | |
|---|---|
| Fichier | `.env`, à la racine (jamais commité) |
| Variable | `VOYAGE_API_KEY` |
| Chargement | `python-dotenv`, appelé au démarrage dans [`config.py`](../src/legalrag/config.py) |
| Priorité | Une variable définie dans l'environnement système l'emporte sur `.env` — c'est ce qui permet d'injecter la clé depuis un CI sans toucher au fichier |

`.env.example` est versionné : il documente les variables attendues, sans
aucune valeur. `.env` ne l'est jamais.

---

## Étape 5 — Vérifier

### 5.1 Ce que la configuration a compris

Cette commande n'affiche **pas** la clé, seulement sa longueur et son
préfixe :

```bash
python -c "
import sys; sys.path.insert(0,'src')
from legalrag.config import load_settings
s = load_settings()
print('modele   :', s.embedding_model, '/', s.embedding_dimension, 'dimensions')
print('endpoint :', s.voyage_base_url)
print('cle      :', len(s.voyage_api_key), 'caracteres, prefixe', repr(s.voyage_api_key[:3]))
"
```

Attendu :

```
modele   : voyage-context-4 / 1024 dimensions
endpoint : https://api.voyageai.com/v1
cle      : 46 caracteres, prefixe 'pa-'
```

Vérifier que l'endpoint correspond bien au préfixe, d'après le tableau de
l'étape 3.

### 5.2 Que l'API répond

C'est le test le plus utile : une requête minuscule qui sépare en une
seconde un problème de clé d'un problème de débit.

```bash
python -c "
from dotenv import dotenv_values
import requests
cle = dotenv_values('.env')['VOYAGE_API_KEY']
r = requests.post(
    'https://api.voyageai.com/v1/contextualizedembeddings',
    headers={'Authorization': f'Bearer {cle}'},
    json={'inputs': [['Article premier du code.']], 'model': 'voyage-context-4',
          'input_type': 'document', 'output_dimension': 1024},
    timeout=60)
print('HTTP', r.status_code)
print(r.json().get('usage') if r.status_code == 200 else r.text[:300])
"
```

Attendu :

```
HTTP 200
{'total_tokens': 5}
```

> Adapter l'URL si votre clé n'est pas préfixée `pa-`, d'après le tableau de
> l'étape 3.

### 5.3 Estimer le travail sans rien payer

```bash
python -m legalrag.cli index --dry-run
```

Liste le corpus, le découpe, et affiche le coût estimé — **sans effectuer le
moindre appel facturé**.

---

## Plafonds de débit

**C'est la section à ne pas sauter.** Elle ne concerne pas l'argent, mais le
temps.

Un compte **sans moyen de paiement enregistré** est plafonné à :

> **3 requêtes et 10 000 tokens par minute**

C'est très bas. Deux conséquences, gérées par le pipeline mais qu'il vaut
mieux comprendre :

1. **La taille d'un lot est bornée par le débit autorisé**, pas seulement
   par la limite d'un appel. Un lot plus gros que le quota d'une minute
   serait refusé à chaque tentative, et **aucun réessai ne pourrait le
   sauver**.
2. **Les chunks d'un même document sont eux aussi découpés selon ce
   budget.** Ils doivent voyager ensemble pour partager leur contexte, donc
   sur un compte plafonné, la fenêtre de contextualisation est plus étroite.

### Réglage recommandé

Dans `.env`, avec une marge sous le plafond annoncé :

```
MAX_REQUESTS_PER_MINUTE=2
MAX_TOKENS_PER_MINUTE=8000
```

À ce rythme, un corpus de ~277 000 tokens prend **une trentaine de
minutes**.

### Pour aller plus vite

Enregistrer un moyen de paiement sur
[dashboard.voyageai.com](https://dashboard.voyageai.com/) débloque les
limites standard **sans rien facturer** : les 200 M de tokens offerts
continuent de s'appliquer.

Passer alors à :

```
MAX_REQUESTS_PER_MINUTE=200
MAX_TOKENS_PER_MINUTE=800000
```

Le même corpus tombe à quelques minutes. Les limites mettent quelques
minutes à s'appliquer après l'ajout du moyen de paiement.

---

## Coût

`voyage-context-4` est facturé **0,12 $ par million de tokens**, avec
**200 M de tokens offerts**.

| Corpus | Tokens | Coût |
|---|---|---|
| ~530 pages | ~277 000 | 0,03 $ |
| ~10 000 pages | ~7 M | 0,84 $ |
| ~100 000 pages | ~70 M | 8,40 $ |

Aucun de ces volumes ne sort du palier gratuit. **Le budget n'est pas le
facteur limitant — le débit l'est.**

---

## Diagnostic

### 401 ou 403

Clé invalide, ou clé envoyée au mauvais domaine. Vérifier le préfixe contre
le tableau de l'étape 3, puis relancer la vérification 5.1.

### 429 avec « You have not yet added your payment method »

C'est le plafond du palier gratuit, pas une erreur de configuration.
Baisser `MAX_REQUESTS_PER_MINUTE` et `MAX_TOKENS_PER_MINUTE` comme indiqué
plus haut, ou enregistrer un moyen de paiement.

### 429 qui persiste malgré les réessais

Un lot plus gros que le quota d'une minute ne passera **jamais**, quel que
soit le nombre de tentatives. Regarder dans les logs la taille annoncée :

```
INFO  legalrag.embeddings  Lot 1 : 1 document(s), 9 chunk(s), ~7200 tokens
```

Ce nombre doit rester sous `MAX_TOKENS_PER_MINUTE`.

### L'indexation est très lente

C'est le comportement attendu sur le palier gratuit. Le limiteur côté client
lisse le débit volontairement, plutôt que de déclencher des 429 en rafale et
de passer son temps en attente de réessai.

### Le job s'est interrompu

Relancer exactement la même commande. Les lots déjà écrits sont conservés et
**ne sont pas repayés** : le pipeline relit les identifiants déjà présents
dans l'index et ne soumet que ce qui manque.

### « Modele inconnu de ce projet »

Le nom dans `EMBEDDING_MODEL` n'est pas dans la liste des modèles connus de
[`config.py`](../src/legalrag/config.py). Le job continue, mais sans
validation du mode d'appel ni des dimensions.

---

## Sécurité

| Règle | Pourquoi |
|---|---|
| Aucune clé en dur dans le code | Tout passe par des variables d'environnement |
| `.env` est dans `.gitignore` | Un secret commité est un secret exposé |
| `.env.example` ne contient aucune valeur | Il documente, il ne transporte pas |
| Les commandes de vérification n'affichent que longueur et préfixe | On peut les coller dans un ticket sans risque |

### Si une clé a fuité

La **révoquer** — dans Atlas (*Model API Keys*) ou sur le dashboard Voyage —
puis en générer une nouvelle.

Retirer la clé de l'historique git ne suffit pas : elle a déjà été exposée.

---

## Voir aussi

- [README](../README.md) — vue d'ensemble du pipeline, et le comparatif des
  modèles d'embedding qui a conduit au choix de `voyage-context-4`
- [Configurer l'accès Google Drive](configuration-google-drive.md) — l'autre
  moitié de la configuration
- [Skill `connexion-voyage`](../.claude/skills/connexion-voyage/SKILL.md) —
  la même procédure, sous une forme exploitable par Claude Code
