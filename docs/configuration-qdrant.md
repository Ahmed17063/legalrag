# Configurer Qdrant

Guide pour stocker l'index d'embeddings dans une base vectorielle.

Le Parquet reste la **source de vérité** : c'est lui qui porte le texte, les
métadonnées et les vecteurs, et c'est lui qu'on rejoue. Qdrant est un cache
d'interrogation, reconstructible à tout moment sans repasser par l'API
d'embeddings — donc sans rien repayer.

---

## Choisir une cible

Trois modes, sélectionnés par la seule variable `QDRANT_URL`.

| Mode | Quand | Prérequis |
|---|---|---|
| **Embarqué** (défaut) | Développement, corpus modeste | Aucun |
| **Serveur local** | Plusieurs processus, Docker disponible | Docker |
| **Qdrant Cloud** | Partagé, accessible à distance | Compte gratuit |

### Mode embarqué

Qdrant écrit dans un répertoire local, sans serveur ni Docker :

```
QDRANT_URL=
QDRANT_PATH=./data/qdrant
```

C'est le mode par défaut, celui qui fonctionne sans rien installer.

Deux limites à connaître :

- **Un seul processus à la fois.** Le répertoire est verrouillé ; une
  seconde commande échouerait tant que la première n'a pas fermé.
- **Les index de payload sont ignorés.** Le filtrage par document ou par
  page fonctionne quand même, mais sans accélération. Sans conséquence à
  quelques centaines de points ; à surveiller au-delà de quelques dizaines
  de milliers.

### Serveur local

```bash
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/data/qdrant-server:/qdrant/storage" qdrant/qdrant
```

```
QDRANT_URL=http://localhost:6333
```

---

## Qdrant Cloud, pas à pas

### Étape 1 — Créer le compte et le cluster

1. Aller sur [cloud.qdrant.io](https://cloud.qdrant.io) et créer un compte
2. **Create cluster**
3. Choisir le plan **Free** — **aucune carte bancaire requise**
4. Choisir un fournisseur (AWS, GCP ou Azure) et une région. Prendre la plus
   proche de là où tourne le job
5. Nommer le cluster, par exemple `legalrag`
6. **Create**

Le cluster met une à deux minutes à démarrer.

**Le palier gratuit** offre 0,5 vCPU, 1 Go de RAM et 4 Go de disque, à
titre permanent. C'est très largement suffisant ici :

| | |
|---|---|
| 348 points × 1024 dimensions × 4 octets | **1,4 Mo** de vecteurs |
| Index Parquet complet, texte compris | 2,8 Mo |
| Capacité du palier gratuit | ~1 million de vecteurs |

> ⚠️ **Un cluster gratuit est suspendu après une semaine d'inactivité, et
> supprimé après quatre semaines.** Sans conséquence ici — le Parquet permet
> de tout recharger en quelques secondes — mais ne comptez pas sur Qdrant
> Cloud gratuit comme stockage durable et unique.

### Étape 2 — Récupérer l'URL et la clé d'API

Dans la console, sur la page du cluster :

- **URL** — de la forme
  `https://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.eu-central-1-0.aws.cloud.qdrant.io:6333`

  Noter le **port `:6333`** : c'est l'API REST. Le port `6334` est le gRPC,
  que ce pipeline n'utilise pas.

- **Clé d'API** — *API Keys* → **Create** → copier immédiatement.

  Comme pour Voyage, la clé n'est affichée qu'une fois.

### Étape 3 — Renseigner `.env`

```
QDRANT_COLLECTION=legalrag
QDRANT_URL=https://xxxxxxxx-....aws.cloud.qdrant.io:6333
QDRANT_API_KEY=<la clé copiée>
QDRANT_BATCH=128
```

`QDRANT_PATH` devient sans effet dès que `QDRANT_URL` est renseignée.

`QDRANT_API_KEY` est couvert par le `.gitignore` au même titre que le reste
de `.env`.

### Étape 4 — Charger

```powershell
.\.venv\Scripts\python.exe -m legalrag.cli qdrant
```

La commande affiche la cible avant de charger, ce qui permet de vérifier
qu'on ne se trompe pas d'instance :

```
Cible                         : serveur https://xxx.cloud.qdrant.io:6333 (avec cle d'API), collection 'legalrag'
Collection creee              : oui
Points charges                : 348
Points dans la collection     : 348
```

La clé d'API n'est jamais affichée, seulement sa présence.

---

## Les commandes

```powershell
# Charger ou mettre à jour
python -m legalrag.cli qdrant

# Repartir d'une collection vide
python -m legalrag.cli qdrant --recreate

# Charger, puis supprimer les points qui ne sont plus dans l'index
python -m legalrag.cli qdrant --prune
```

**Le rechargement est idempotent.** L'identifiant d'un point dérive du
`chunk_id` par UUID déterministe : recharger deux fois met à jour les mêmes
points au lieu d'en créer des doublons. On peut donc relancer sans crainte.

`--prune` est le pendant de celui de l'indexation : si le corpus a changé,
d'anciens points subsistent dans Qdrant sans plus correspondre à rien, et
remonteraient à la recherche.

---

## Ce qui est stocké

Chaque point porte le vecteur et, en payload, toutes les métadonnées :

| Champ | Usage |
|---|---|
| `chunk_id` | Identité, élagage |
| `text` | Le passage lui-même, à citer dans la réponse |
| `document`, `document_id` | Filtrer sur un code précis |
| `page_start`, `page_end`, `pages` | Citer la source avec précision |
| `timestamp` | Distinguer deux versions d'un document |
| `source_filenames`, `chunk_index`, `token_count` | Traçabilité |

Quatre champs sont indexés pour le filtrage — `document_id`, `timestamp`
(mot-clé), `page_start`, `page_end` (entier) — ce qui permettra de
restreindre une recherche à un code ou à une plage de pages.

La distance configurée est le **cosinus**. Les vecteurs Voyage sont déjà
normés, donc le produit scalaire donnerait le même classement ; le cosinus
reste correct si la source cesse un jour de l'être.

---

## Vérifier

### Le contenu de la collection

```bash
python -c "
import sys; sys.path.insert(0,'src')
from qdrant_client import QdrantClient
from dotenv import dotenv_values
env = dotenv_values('.env')
c = (QdrantClient(url=env['QDRANT_URL'], api_key=env.get('QDRANT_API_KEY'))
     if env.get('QDRANT_URL') else QdrantClient(path=env.get('QDRANT_PATH','./data/qdrant')))
try:
    print('points :', c.count(env.get('QDRANT_COLLECTION','legalrag'), exact=True).count)
finally:
    c.close()
"
```

### Une vraie recherche

Elle nécessite un appel d'embedding pour la question — un seul, quelques
tokens. Voir [configuration-voyage.md](configuration-voyage.md) pour la
forme exacte de la requête, avec `input_type="query"` et non `"document"`.

---

## Diagnostic

**`Storage folder ... is already accessed by another instance`** — mode
embarqué, deux processus. Fermer l'autre, ou passer sur un serveur.

**403 / `Invalid api key`** — clé erronée, ou clé d'un autre cluster.

**Timeout à la connexion** — vérifier que l'URL se termine par `:6333` et
non `:6334` (gRPC).

**`La collection attend des vecteurs de N dimensions`** — la collection a
été créée avec un autre modèle d'embedding. Mélanger deux modèles rendrait
les distances incomparables : utiliser `--recreate`, ou une autre
collection.

**Cluster gratuit inaccessible après quelques jours** — il a été suspendu
pour inactivité. Le réactiver dans la console, puis recharger depuis le
Parquet.

**`Aucun index trouve dans data/index`** — lancer d'abord
`python -m legalrag.cli index`.

---

## Voir aussi

- [README](../README.md) — vue d'ensemble du pipeline
- [Configurer l'accès Google Drive](configuration-google-drive.md)
- [Configurer la clé Voyage](configuration-voyage.md)
