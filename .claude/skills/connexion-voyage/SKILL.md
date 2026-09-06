---
name: connexion-voyage
description: Obtenir et configurer une clé d'API Voyage AI pour les embeddings, choisir le bon endpoint selon l'origine de la clé, et régler les plafonds de débit. À utiliser quand l'utilisateur configure sa clé Voyage, obtient une erreur 401, 403 ou 429, ou trouve l'indexation anormalement lente.
---

# Connexion à Voyage AI

## Obtenir une clé

Voyage AI appartient à MongoDB. Il existe **deux parcours**, et ils ne
donnent pas accès au même domaine.

### Parcours actuel — MongoDB Atlas

Aucun cluster payant ni carte bancaire requis, et il n'est pas nécessaire
d'être client de la base MongoDB.

1. Compte sur [cloud.mongodb.com](https://cloud.mongodb.com) — Atlas crée
   une organisation et un projet automatiquement.
2. Barre de navigation, sous **Services** → **Model API Keys**.
3. **Create new model API key**, un nom, sélectionner le projet.
4. Laisser **Cloud provider** et **Geography** sur `any`. Restreindre une
   clé à une zone impose un palier payant et oblige à viser un endpoint
   différent.
5. **Create**, puis **copier la clé immédiatement**.

La clé n'est **plus jamais réaffichée**. En cas de perte, en créer une
autre.

### Parcours historique — dashboard Voyage

Les comptes créés sur [dashboard.voyageai.com](https://dashboard.voyageai.com/)
délivrent des clés préfixées `pa-`. Elles restent valides.

## Endpoint : le piège

Envoyer une clé au mauvais domaine échoue à l'authentification, avec un
message qui n'indique pas la cause.

| Origine | Préfixe | Endpoint |
|---|---|---|
| Atlas, non restreinte | aucun | `ai.mongodb.com` |
| Atlas, Europe | `al-eu-` | `eu.ai.mongodb.com` |
| Atlas, États-Unis | `al-us-` | `us.ai.mongodb.com` |
| Ancienne plateforme | `pa-` | `api.voyageai.com` |

Le pipeline déduit l'endpoint du préfixe
(`legalrag.config.resolve_base_url`), il n'y a normalement rien à faire.
`VOYAGE_API_BASE_URL` force une URL si nécessaire.

## Où mettre la clé

Dans `.env` à la racine (ignoré par git) :

```
VOYAGE_API_KEY=<la clé copiée>
VOYAGE_API_BASE_URL=
```

Laisser `VOYAGE_API_BASE_URL` vide : l'URL se déduit du préfixe.

Une variable déjà définie dans l'environnement l'emporte sur `.env`, ce qui
permet d'injecter la clé par le CI sans toucher au fichier.

## Vérifier

Contrôler ce que la configuration a compris, sans afficher la clé :

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

Puis tester l'API avec une requête minuscule. C'est le moyen le plus rapide
de séparer un problème de clé d'un problème de débit :

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
print('HTTP', r.status_code, r.json().get('usage') if r.status_code == 200 else r.text[:300])
"
```

Adapter l'URL si la clé n'est pas préfixée `pa-`.

## Plafonds de débit

**Un compte sans moyen de paiement enregistré est plafonné à 3 requêtes et
10 000 tokens par minute.** C'est très bas et c'est la cause la plus
fréquente des 429.

Deux conséquences, gérées par le pipeline mais qu'il faut comprendre :

- la taille d'un lot est bornée par le débit autorisé, pas seulement par la
  limite d'un appel. Un lot plus gros que le quota d'une minute serait
  refusé à chaque tentative, et **aucun réessai ne pourrait le sauver** ;
- les chunks d'un même document, qui doivent voyager ensemble pour partager
  leur contexte, sont eux aussi découpés selon ce budget. Sur un compte
  plafonné, la fenêtre de contextualisation est plus étroite.

Réglage dans `.env`, avec une marge sous le plafond annoncé :

```
MAX_REQUESTS_PER_MINUTE=2
MAX_TOKENS_PER_MINUTE=8000
```

Enregistrer un moyen de paiement débloque les limites standard **sans rien
facturer** : les 200 M de tokens offerts continuent de s'appliquer. Passer
alors à `200` et `800000`.

## Diagnostic

**401 / 403** — clé invalide, ou clé envoyée au mauvais domaine (voir le
tableau des endpoints). Vérifier le préfixe.

**429 avec « You have not yet added your payment method »** — plafond du
palier gratuit. Baisser `MAX_REQUESTS_PER_MINUTE` et
`MAX_TOKENS_PER_MINUTE`, ou enregistrer un moyen de paiement.

**429 persistant malgré des réessais** — un lot plus gros que le quota
d'une minute ne passera jamais. Vérifier dans les logs la taille annoncée
(`Lot N : ... ~X tokens`) : elle doit rester sous
`MAX_TOKENS_PER_MINUTE`.

**Indexation lente** — c'est le comportement attendu sur le palier gratuit.
Le limiteur côté client lisse le débit plutôt que de déclencher des 429 en
rafale. Compter une trentaine de minutes pour ~277 000 tokens à 8 000/min.

**Job interrompu** — relancer la même commande. Les lots déjà écrits sont
conservés et ne sont pas repayés.

## Coût

Le tarif `voyage-context-4` est de 0,12 $ par million de tokens, avec
200 M offerts. Un corpus de quelques centaines de pages coûte quelques
centimes et reste entièrement dans le palier gratuit. Le budget n'est pas
le facteur limitant — le débit l'est.

Estimer avant de lancer, sans facturer :

```bash
python -m legalrag.cli index --dry-run
```
