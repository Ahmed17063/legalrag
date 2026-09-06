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

```bash
uv venv --python 3.11
uv pip install -r requirements-dev.txt
```

## 3. Où mettre la clé d'API

Aucune clé n'est écrite dans le code. Tout passe par l'environnement.

```bash
cp .env.example .env
```

Puis dans `.env` :

```
VOYAGE_API_KEY=pa-xxxxxxxxxxxxxxxxxxxxxxxx
```

| | |
|---|---|
| Fichier à créer | `.env`, à la racine du dépôt |
| Variable | `VOYAGE_API_KEY` |
| Chargement | `python-dotenv`, appelé dans [`config.py`](src/legalrag/config.py) au démarrage |
| Priorité | Une variable déjà définie dans l'environnement l'emporte sur `.env` — c'est ce qui permet d'injecter la clé par le CI sans toucher au fichier |

La clé s'obtient sur le [dashboard Voyage](https://dashboard.voyageai.com/)
→ *API Keys*.

`.gitignore` couvre déjà :

```
.env
.env.*
!.env.example
secrets/
*service-account*.json
*credentials*.json
data/
```

`.env.example` est versionné (il documente les variables attendues, sans
valeurs) ; `.env` ne l'est jamais. Si une clé a été commitée par accident,
la révoquer sur le dashboard : la retirer de l'historique git ne suffit pas,
elle a déjà fuité.

## 4. Credentials Google Drive

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

## 5. Utilisation

```bash
# Ce qui serait fait, et ce que ça coûterait — aucun appel facturé
python -m legalrag.cli index --dry-run

# Sur un échantillon, pour valider la chaîne de bout en bout
python -m legalrag.cli index --limit 20

# Le job complet, puis fusion des fragments en un fichier unique
python -m legalrag.cli index --compact

# Fusionner plus tard un index déjà construit
python -m legalrag.cli compact --remove-parts
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

Étape suivante naturelle : charger ce Parquet dans un Qdrant (déjà présent
dans l'ancienne version du projet). Le Parquet reste la source de vérité
rejouable, la base vectorielle devient un cache d'interrogation.

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
- **on change de modèle ou de dimension** → l'index refuse de s'ouvrir,
  plutôt que de mélanger des vecteurs incomparables. Utiliser un `INDEX_DIR`
  distinct.

## 8. Organisation du code

| Fichier | Rôle |
|---|---|
| [`filename_parser.py`](src/legalrag/filename_parser.py) | Nom de fichier → document / page / horodatage |
| [`drive.py`](src/legalrag/drive.py) | Listing et téléchargement Drive, cache, retry |
| [`tokenization.py`](src/legalrag/tokenization.py) | Comptage de tokens et positions caractères |
| [`chunking.py`](src/legalrag/chunking.py) | Recollement des pages et découpage avec chevauchement |
| [`embeddings.py`](src/legalrag/embeddings.py) | Appels API par lots, rate limiting, retry |
| [`index_store.py`](src/legalrag/index_store.py) | Écriture Parquet atomique et reprise |
| [`pipeline.py`](src/legalrag/pipeline.py) | Orchestration |
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

## 9. Tests

```bash
python -m pytest
```

67 tests, sans appel réseau. Couvrent le parsing des noms (avec et sans
horodatage, titres à tirets, pages à plusieurs chiffres, fichiers hors
convention), le chunking (reconstitution d'un article coupé entre deux
pages, chevauchement effectif, pages non contiguës), le découpage en lots et
l'appariement chunk ↔ vecteur, et la reprise après incident.

## 10. Ce qui n'est pas fait

- La **recherche / le retrieval** : hors périmètre de cette étape.
- L'**évaluation de la qualité** de restitution sur ton corpus, qui reste la
  seule façon de trancher définitivement le choix du modèle (voir §1).
