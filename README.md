# Indexation d'un corpus juridique français

Pipeline d'embeddings pour un RAG sur des codes et textes de loi français.
Le corpus vit sur Google Drive sous forme de `.txt`, un fichier par page.

**Périmètre de cette étape : de la lecture des documents à la production de
l'index d'embeddings.** La recherche / le retrieval n'est pas implémenté ici.

---

## 1. Choix de l'API d'embedding

### Comparaison

Tarifs et spécifications relevés le 6 septembre 2026.

| Modèle | $ / M tokens | Dimensions | Entrée max | Français juridique |
|---|---|---|---|---|
| **`voyage-context-4`** | **0,12** | 256 / 512 / 1024 / 2048 | 32 000 tok | Multilingue. Embeddings *contextualisés* : chaque chunk est plongé en tenant compte du document entier. En tête sur le domaine `law` dans 5 des 8 domaines évalués par l'éditeur |
| `voyage-law-2` | 0,12 | 1024 (fixe) | 16 000 tok | Spécialisé juridique, mais génération 2024, contexte deux fois plus court, pas de contextualisation |
| `text-embedding-3-large` (OpenAI) | 0,13 | 3072 (réductible) | 8 191 tok | Généraliste multilingue solide, aucune spécialisation juridique. Contexte le plus court du lot |
| `mistral-embed` | ~0,10 | 1024 | 8 000 tok | Éditeur français, français généraliste correct. Tarif à confirmer : les sources publiques divergent entre 0,06 et 0,10 |

### Recommandation : `voyage-context-4`

Trois raisons, par ordre d'importance pour ce corpus précis.

**1. Il répond directement au problème de la pagination.** Ton corpus est
découpé en pages, et un article de code coupé en fin de page produit deux
fragments incomplets. Un modèle d'embedding classique plonge chaque chunk
isolément : le fragment « ...est puni de trois ans d'emprisonnement » perd
tout lien avec l'article qui l'introduit. `voyage-context-4` reçoit les
chunks d'un document groupés et produit un vecteur par chunk *qui tient
compte du reste du document*. Le chevauchement entre pages, que ce pipeline
implémente par ailleurs, cesse d'être la seule ligne de défense.

**2. Le coût est négligeable à ton volume.** À 0,12 $/M tokens, et avec
**200 M de tokens offerts** sur les modèles de la série 4, un corpus de
10 000 pages (~600 tokens par page, ~20 % de recouvrement, soit ~7 M tokens)
tient entièrement dans le palier gratuit. Même à 100 000 pages tu es à
environ 8 $. Le budget n'est pas le critère discriminant ici — la qualité de
restitution l'est.

**3. Les dimensions sont ajustables** (Matryoshka : 256 à 2048). Tu peux
démarrer en 1024 et redescendre à 512 si le stockage ou la latence de
recherche devient un sujet, sans changer de modèle.

**Pourquoi pas `voyage-law-2`**, pourtant explicitement « juridique » : c'est
un modèle de 2024, à contexte deux fois plus court et sans contextualisation.
Sa spécialisation de domaine ne compense pas, à mon avis, un écart de deux
générations — d'autant que sa spécialisation a été construite sur du droit
majoritairement anglophone (*common law*), pas sur du droit civil français.

**Réserve à connaître.** Les chiffres de performance ci-dessus sont ceux
publiés par l'éditeur, sur ses propres jeux d'évaluation, et je n'ai trouvé
aucun benchmark indépendant sur du **français juridique** spécifiquement. La
recommandation repose donc sur des arguments structurels (contextualisation,
contexte long, multilinguisme) plus que sur une mesure. Avant de figer le
choix, constitue une cinquantaine de questions-réponses de référence sur ton
corpus et compare `voyage-context-4` et `mistral-embed` en conditions
réelles. Le pipeline permet d'en changer par une variable d'environnement, et
l'index refuse de mélanger deux modèles.

---

## 2. Installation

Les dépendances sont gérées par [uv](https://docs.astral.sh/uv/).
`pyproject.toml` les déclare, `uv.lock` les verrouille — les deux sont
versionnés, donc l'installation est reproductible à l'identique.

```bash
uv sync --all-groups
```

C'est tout : uv crée le `.venv`, installe les dépendances **et** le projet
lui-même en mode éditable. Aucune activation n'est nécessaire avec `uv run` :

```bash
uv run legalrag ask "ta question"
uv run pytest
```

| Commande | Effet |
|---|---|
| `uv sync` | Dépendances de production seulement |
| `uv sync --all-groups` | Y compris `dev` (pytest) |
| `uv add <paquet>` | Ajoute une dépendance et met `uv.lock` à jour |
| `uv add --group dev <paquet>` | Idem, dans le groupe de développement |
| `uv lock --upgrade` | Remonte les versions dans les bornes déclarées |
| `uv run <commande>` | Exécute dans l'environnement, sans l'activer |

`requirements.txt` n'existe plus : `pyproject.toml` est la source unique. Si
un contexte l'exige malgré tout (image Docker minimale, CI qui ne connaît pas
uv) :

```bash
uv export --no-dev --format requirements-txt > requirements.txt
```

Ne pas versionner ce fichier généré — il divergerait du lock.

### Dans VS Code

[.vscode/settings.json](.vscode/settings.json) pointe l'interpréteur sur le
`.venv` du projet. L'extension Python active l'environnement dans chaque
nouveau terminal intégré, ce qui permet d'écrire directement :

```powershell
uv run legalrag ask "ta question"
```

Le réglage est limité au dossier : le PATH global de la machine n'est pas
modifié, les autres projets ne sont pas affectés.

## 3. Clé d'API Voyage

> Guide détaillé, écran par écran :
> **[docs/configuration-voyage.md](docs/configuration-voyage.md)**

### L'obtenir

Voyage AI appartient à MongoDB. Les clés se créent désormais dans MongoDB
Atlas — **aucun cluster payant ni carte bancaire n'est requis**, et il n'est
pas nécessaire d'être client de la base de données MongoDB.

1. Créer un compte sur [cloud.mongodb.com](https://cloud.mongodb.com).
   Atlas crée automatiquement une organisation et un projet.
2. Dans la barre de navigation, sous **Services**, cliquer sur
   **Model API Keys**.
3. **Create new model API key**, lui donner un nom, sélectionner le projet.
4. Laisser **Cloud provider** et **Geography** sur `any` : restreindre une
   clé à une zone impose un palier payant, et oblige à viser un endpoint
   différent.
5. **Create**, puis **copier la clé immédiatement**. Elle n'est plus jamais
   affichée ensuite — en cas de perte, il faut en créer une autre.

Le palier gratuit couvre 200 M de tokens sur la série `voyage-4`
(50 M sur `voyage-law-2`).

### Endpoint : attention au piège

Il existe deux domaines, selon l'origine de la clé :

| Origine de la clé | Préfixe | Endpoint |
|---|---|---|
| MongoDB Atlas, non restreinte | aucun | `ai.mongodb.com` |
| MongoDB Atlas, Europe | `al-eu-` | `eu.ai.mongodb.com` |
| MongoDB Atlas, États-Unis | `al-us-` | `us.ai.mongodb.com` |
| Ancien compte Voyage | `pa-` | `api.voyageai.com` |

Envoyer une clé au mauvais domaine échoue à l'authentification. Le pipeline
déduit l'endpoint du préfixe de la clé
([`resolve_base_url`](src/legalrag/config.py)), il n'y a donc normalement
rien à configurer. `VOYAGE_API_BASE_URL` permet de forcer une URL si besoin.

### Où la mettre

```bash
cp .env.example .env
```

Puis dans `.env` :

```
VOYAGE_API_KEY=<la clé copiée>
```

| | |
|---|---|
| Fichier | `.env`, à la racine du dépôt |
| Variable | `VOYAGE_API_KEY` |
| Chargement | `python-dotenv`, appelé dans [`config.py`](src/legalrag/config.py) |
| Priorité | Une variable déjà définie dans l'environnement l'emporte sur `.env` — c'est ce qui permet d'injecter la clé par le CI sans toucher au fichier |

`.gitignore` couvre déjà `.env`, `secrets/`, `*service-account*.json`,
`*credentials*.json` et `data/`. `.env.example` est versionné (il documente
les variables attendues, sans valeurs) ; `.env` ne l'est jamais.

Si une clé a été commitée par accident, la révoquer dans Atlas : la retirer
de l'historique git ne suffit pas, elle a déjà fuité.

## 4. Credentials Google Drive

> Guide détaillé, écran par écran :
> **[docs/configuration-google-drive.md](docs/configuration-google-drive.md)**

Le pipeline lit Drive avec un **compte de service** — pas d'OAuth
interactif, donc le job tourne sans navigateur ni token à rafraîchir.

1. **Console Google Cloud** → [console.cloud.google.com](https://console.cloud.google.com/)
   → créer un projet (ou en réutiliser un).

2. **Activer l'API Drive** : *APIs & Services* → *Library* → « Google Drive
   API » → *Enable*.

3. **Créer le compte de service** : *APIs & Services* → *Credentials* →
   *Create credentials* → *Service account*. Un nom suffit ; l'étape
   « rôles » peut être passée (les permissions viennent du partage Drive,
   pas d'IAM).

4. **Générer la clé JSON** : ouvrir le compte de service créé → onglet
   *Keys* → *Add key* → *Create new key* → **JSON**. Le fichier se
   télécharge. Le placer hors du dépôt, ou dans `secrets/` (déjà ignoré) :

   ```bash
   mkdir -p secrets && mv ~/Downloads/xxxx.json secrets/service-account.json
   ```

5. **Partager le dossier Drive avec le compte de service.** C'est l'étape
   qu'on oublie, et elle produit une erreur 404 trompeuse (« fichier
   introuvable » plutôt que « accès refusé »). Récupérer l'adresse du compte
   de service dans le JSON (champ `client_email`, de la forme
   `xxx@yyy.iam.gserviceaccount.com`), puis sur le dossier Drive :
   *Partager* → coller cette adresse → **Lecteur** → *Envoyer*.

6. **Renseigner `.env`** :

   ```
   GOOGLE_APPLICATION_CREDENTIALS=./secrets/service-account.json
   DRIVE_FOLDER_ID=1TwWEQk33CI9M8v_mdj2gpKPoOKariJm2
   ```

   L'ID du dossier est le segment après `/folders/` dans son URL.

Le scope demandé est `drive.readonly` : le pipeline ne peut rien écrire ni
supprimer sur ton Drive, même en cas de bug.

## 5. Débit et plafonds

Un compte Voyage **sans moyen de paiement enregistré** est plafonné à
**3 requêtes et 10 000 tokens par minute**. C'est peu, et cela a deux
conséquences que le pipeline gère :

- la taille d'un lot est bornée par le débit autorisé, pas seulement par la
  limite d'un appel. Un lot plus gros que le quota d'une minute serait
  refusé à chaque tentative, et aucun réessai ne le sauverait ;
- les chunks d'un même document, qui doivent voyager ensemble pour partager
  leur contexte, sont eux aussi découpés selon ce budget. Sur un compte
  plafonné, la fenêtre de contextualisation est donc plus étroite.

Les valeurs par défaut (`MAX_REQUESTS_PER_MINUTE=2`,
`MAX_TOKENS_PER_MINUTE=8000`) gardent une marge sous le plafond annoncé.
Le corpus complet (~277 000 tokens) prend alors une trentaine de minutes.

Avec un moyen de paiement enregistré — **les 200 M de tokens offerts
continuent de s'appliquer**, la facture reste nulle — passer à `200` et
`800000` réduit le job à quelques minutes.

## 5 bis. Utilisation

```bash
# Ce qui serait fait, et ce que ça coûterait — aucun appel facturé
uv run legalrag index --dry-run

# Sur un échantillon, pour valider la chaîne de bout en bout
uv run legalrag index --limit 20

# Le job complet : élagage des lignes obsolètes, puis fusion des fragments
uv run legalrag index --prune --compact

# Fusionner plus tard un index déjà construit
uv run legalrag compact --remove-parts
```

`-v` pour les logs détaillés.

## 6. Format de sortie

**Parquet, en local.** `data/index/` :

```
data/index/
├── manifest.json              modèle, dimension, paramètres de chunking
├── parts/
│   ├── part-000000.parquet    un fragment par lot d'API
│   └── part-000001.parquet
└── embeddings.parquet         après --compact
```

Une ligne par chunk : `chunk_id`, `document`, `document_id`, `chunk_index`,
`text`, `page_start`, `page_end`, `pages`, `timestamp`, `source_filenames`,
`token_count`, `embedding`.

**Pourquoi Parquet** : colonnaire et typé, un vecteur de 1024 `float32` pèse
4 Ko contre ~20 Ko en JSON ; le texte et les métadonnées vivent dans le même
fichier que les vecteurs, donc l'index est auto-portant ; et `pyarrow` sait
ne lire qu'une colonne, ce qui rend la reprise quasi instantanée.

**Pourquoi en local plutôt que sur Drive** : Drive n'a pas d'écriture
atomique ni d'append. Écrire l'index au fil de l'eau y imposerait de
réuploader le fichier entier à chaque lot, sous quota d'API, avec le risque
qu'une interruption laisse un fichier corrompu — exactement ce que la
reprise sur incident cherche à éviter. Le disque local est rapide, atomique,
et le fichier compacté peut être poussé où tu veux **une fois** le job
terminé. Écrire sur Drive obligerait par ailleurs à élargir les droits du
compte de service au-delà de la lecture seule.

## 6 bis. Charger dans Qdrant

```bash
uv run legalrag qdrant           # charger ou mettre à jour
uv run legalrag qdrant --prune   # + supprimer les points obsolètes
uv run legalrag qdrant --recreate
```

Trois cibles, choisies par la seule variable `QDRANT_URL` : embarqué sur
disque (défaut, aucun prérequis), serveur local, ou Qdrant Cloud. Le
rechargement est idempotent — l'identifiant d'un point dérive du `chunk_id`,
donc recharger met à jour au lieu de dupliquer.

Le Parquet reste la source de vérité rejouable ; la base vectorielle est un
cache d'interrogation, reconstructible sans repasser par l'API d'embeddings.

> Guide détaillé, dont Qdrant Cloud :
> **[docs/configuration-qdrant.md](docs/configuration-qdrant.md)**

## 7. Reprise sur incident

`chunk_id` est un hachage du contenu du chunk et de son rang. Au démarrage,
le pipeline relit la colonne `chunk_id` des fragments déjà écrits et ne
soumet à l'API que ce qui manque.

Concrètement :

- **le job meurt à mi-parcours** → relancer la même commande. Ce qui est
  écrit est conservé, rien n'est repayé ;
- **chaque lot est écrit dès son retour d'API**, dans un fichier temporaire
  puis renommé (`os.replace`, atomique). Un fragment est complet ou absent,
  jamais tronqué ;
- **les fichiers Drive sont mis en cache** sous `data/cache/`, indexés par
  identifiant et checksum : une reprise ne retélécharge pas ;
- **le texte source change** → le `chunk_id` change → ce chunk-là est
  replongé, les autres non ;
- **le corpus change** (texte modifié, premier passage lancé avec `--limit`)
  → les anciens chunks restent dans l'index sans plus correspondre à rien.
  `--prune` les supprime : il compare l'index au découpage courant et
  retire ce qui n'a plus d'équivalent. Ignoré avec `--limit`, où tout le
  reste de l'index paraîtrait à tort obsolète ;
- **on change de modèle ou de dimension** → l'index refuse de s'ouvrir,
  plutôt que de mélanger des vecteurs incomparables. Utiliser un `INDEX_DIR`
  distinct.

## 8. Organisation du code

| Fichier | Rôle |
|---|---|
| [`filename_parser.py`](src/legalrag/filename_parser.py) | Nom de fichier → document / page / horodatage |
| [`drive.py`](src/legalrag/drive.py) | Listing et téléchargement Drive, cache, retry |
| [`tokenization.py`](src/legalrag/tokenization.py) | Comptage de tokens et positions caractères |
| [`boilerplate.py`](src/legalrag/boilerplate.py) | Détection et retrait des en-têtes/pieds de page répétés |
| [`chunking.py`](src/legalrag/chunking.py) | Recollement des pages et découpage avec chevauchement |
| [`embeddings.py`](src/legalrag/embeddings.py) | Appels API par lots, rate limiting, retry |
| [`qdrant_store.py`](src/legalrag/qdrant_store.py) | Chargement de l'index dans Qdrant (embarqué, serveur ou Cloud) |
| [`index_store.py`](src/legalrag/index_store.py) | Écriture Parquet atomique et reprise |
| [`pipeline.py`](src/legalrag/pipeline.py) | Orchestration |
| [`retrieval.py`](src/legalrag/retrieval.py) | Recherche : vectoriel, découpage par article, reranking |
| [`generation.py`](src/legalrag/generation.py) | Rédaction de la réponse (Groq), avec vérification des citations |
| [`cli.py`](src/legalrag/cli.py) | Ligne de commande |
| [`config.py`](src/legalrag/config.py) | Configuration par variables d'environnement |

### Le chunking, en deux mots

Les pages **consécutives** d'un même document sont recollées en un texte
continu, puis découpées par fenêtre glissante avec chevauchement. Un article
à cheval sur les pages 20 et 21 se retrouve donc entier dans au moins un
chunk, qui porte `pages = [20, 21]`.

Deux détails qui comptent :

- si une page se termine sans ponctuation finale, la phrase court sur la
  suivante : les deux moitiés sont recollées par une simple espace, sans
  coupure de paragraphe au milieu de l'article ;
- une page manquante coupe la continuité. Si le corpus contient les pages 20
  et 22 mais pas la 21, elles ne sont pas recollées : cela fabriquerait un
  faux voisinage.

### Les bandeaux de page

Les corpus numérisés portent un en-tête et un pied de page sur chaque page
(institution, titre du document). Ils sont retirés **avant** le chunking,
sinon le bandeau s'insère au milieu des articles à cheval sur deux pages —
précisément la coupure que le recollement doit supprimer.

La détection est statistique, pas codée en dur : une ligne est un bandeau si
elle apparaît en tête ou en pied de plus de la moitié des pages du même
document. Le bruit d'OCR (le même bandeau ressort en « ...Justice : LS »,
« ...Justice D », « ...Justice T ») est absorbé en ignorant les jetons courts
de fin de ligne. Les numéros, eux, sont conservés : sans cela « Article 12 »
et « Article 13 » se confondraient et du contenu serait supprimé.

Garde-fous : une ligne ouvrant un article, un titre, un chapitre ou une
section n'est jamais retirée, et un retrait qui viderait une page est annulé.

Mesuré sur le corpus mauritanien : **7,8 % des caractères** retirés,
378 → 348 chunks. `STRIP_BOILERPLATE=false` désactive le mécanisme.

## 9. Tests

```bash
uv run pytest
```

67 tests, sans appel réseau. Couvrent le parsing des noms (avec et sans
horodatage, titres à tirets, pages à plusieurs chiffres, fichiers hors
convention), le chunking (reconstitution d'un article coupé entre deux
pages, chevauchement effectif, pages non contiguës), le découpage en lots et
l'appariement chunk ↔ vecteur, et la reprise après incident.

## 9 bis. Poser une question

```bash
# Les passages sourcés, sans rédaction
uv run legalrag search "ta question" --top-k 8 --top-n 3

# Une réponse rédigée, avec ses sources
uv run legalrag ask "ta question" --top-k 8 --top-n 3
uv run legalrag ask "..." --show-passages   # + le texte intégral cité
```

La recherche procède en trois temps : vectoriel dans Qdrant, découpage de
chaque chunk sur ses en-têtes d'article, puis reranking des passages face à
la question. Sans ce découpage, un chunk de 2554 caractères contenant
6,3 articles en moyenne noierait l'article pertinent.

### Les garde-fous de la génération

Sur du juridique, un modèle qui reformule approximativement une disposition
produit une réponse plausible et fausse — pire qu'inutile. Trois protections,
toutes vérifiables :

- **passages numérotés, citation obligatoire** : chaque affirmation doit
  porter son `[n]` ;
- **température nulle** : aucune raison de vouloir de la variété ;
- **vérification après coup** : les numéros cités sont contrôlés. Hors plage,
  ils sont signalés ; une réponse sans citation est marquée non étayée.

La réponse est toujours rendue **avec** les passages qui l'ont nourrie : la
vérification humaine doit rester possible. Le modèle a pour consigne d'écrire
exactement « La réponse ne figure pas dans les documents fournis » plutôt que
d'inventer.

## 10. Skills Claude Code

Deux skills du dépôt documentent la mise en place des accès, avec les
commandes de diagnostic et les pièges rencontrés :

| Skill | Couvre |
|---|---|
| [`connexion-drive`](.claude/skills/connexion-drive/SKILL.md) | Compte de service, partage du dossier, « 0 fichier trouvé », 403/404 |
| [`connexion-voyage`](.claude/skills/connexion-voyage/SKILL.md) | Clé d'API, choix de l'endpoint selon le préfixe, plafonds de débit, 401/429 |

Les invoquer avec `/connexion-drive` ou `/connexion-voyage`.

## 11. Ce qui n'est pas fait

- La **recherche / le retrieval** : hors périmètre de cette étape.
- L'**évaluation de la qualité** de restitution sur ton corpus, qui reste la
  seule façon de trancher définitivement le choix du modèle (voir §1).
