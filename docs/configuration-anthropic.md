# Configurer la clé d'API Anthropic

Guide pas à pas pour activer **Claude Sonnet 5** comme second modèle
rédacteur, à côté de Groq.

**Durée : environ 5 minutes.** Une carte bancaire est nécessaire — contrairement
à Voyage et à Groq, Anthropic n'offre pas de palier gratuit sur l'API. Voir
[Coût](#coût) pour les montants réels, qui sont faibles pour cet usage.

---

## Ce qu'on va faire

```
┌──────────────────┐       ┌──────────────┐       ┌────────────────────┐
│  Compte console  │       │  Crédits     │       │  Clé d'API         │
│  Anthropic       │──────▶│  prépayés    │──────▶│  (secrète,         │
└──────────────────┘ crée  │  (5 $ suffit)│ puis  │  affichée une fois)│
                           └──────────────┘       └────────────────────┘
                                                            │
                                                            ▼
                                                 .env : ANTHROPIC_API_KEY
```

Trois idées à retenir :

1. **La clé n'est affichée qu'une seule fois.** Il faut la copier
   immédiatement ; sinon, il faut en générer une autre.
2. **Sans crédits, la clé existe mais ne répond pas.** L'appel échoue avec
   un message de facturation, pas d'authentification — c'est déroutant si on
   ne s'y attend pas.
3. **Ce n'est pas le même compte que claude.ai.** Un abonnement Claude Pro ou
   Max ne donne aucun accès à l'API, et les crédits d'API ne donnent aucun
   accès à claude.ai. Ce sont deux produits facturés séparément.

---

## Étape 1 — Créer un compte sur la console

1. Aller sur **<https://console.anthropic.com>**
2. Créer un compte (e-mail, ou connexion Google)
3. Vérifier l'adresse e-mail
4. Anthropic crée automatiquement une organisation

> Si vous avez déjà un compte **claude.ai**, la même adresse fonctionne pour
> vous connecter, mais l'organisation d'API reste distincte de l'abonnement.

---

## Étape 2 — Acheter des crédits

C'est l'étape qu'on oublie, et qui fait échouer le premier appel.

1. Dans la console, ouvrir **Settings → Billing**
   (<https://console.anthropic.com/settings/billing>)
2. Cliquer sur **Add payment method** et renseigner une carte
3. Cliquer sur **Buy credits** et acheter le minimum proposé — **5 $ suffisent
   très largement** pour cet usage (voir [Coût](#coût))

Facultatif mais recommandé : désactiver le rechargement automatique
(*auto-reload*) tant que vous n'avez pas mesuré votre consommation réelle.
Cela transforme un éventuel emballement en simple interruption de service
plutôt qu'en facture.

---

## Étape 3 — Générer la clé

1. Ouvrir **Settings → API keys**
   (<https://console.anthropic.com/settings/keys>)
2. Cliquer sur **Create Key**
3. Lui donner un nom qui vous parlera dans six mois — par exemple
   `chatbot-juridique-mauritanie`
4. Choisir le *workspace* `Default` si la question est posée
5. Cliquer sur **Add** ou **Create**

La clé s'affiche alors **une seule fois**. Elle ressemble à :

```
sk-ant-api03-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

**Cliquer sur « Copy » immédiatement.** Une fois la fenêtre fermée, la clé
n'est plus récupérable : il faut en créer une autre et supprimer l'ancienne.

---

## Étape 4 — Coller la clé dans `.env`

Ouvrir le fichier `.env` à la racine du projet. La ligne existe déjà, vide :

```dotenv
ANTHROPIC_API_KEY=
```

Coller la clé juste après le signe égal :

```dotenv
ANTHROPIC_API_KEY=sk-ant-api03-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Quatre pièges classiques, tous silencieux :

| À ne pas faire | Pourquoi |
|---|---|
| `ANTHROPIC_API_KEY="sk-ant-..."` | Les guillemets sont pris pour une partie de la clé |
| `ANTHROPIC_API_KEY = sk-ant-...` | Les espaces autour du `=` cassent la lecture du fichier |
| Coller dans `.env.example` | Ce fichier est versionné : la clé partirait sur GitLab |
| Laisser un retour à la ligne au milieu | Une clé collée depuis un PDF peut être coupée |

`.env` est ignoré par git (`.gitignore`, ligne 49) : la clé ne partira pas
dans un commit.

---

## Étape 5 — Vérifier

### 5.1 La configuration est-elle lue ?

```bash
uv run python -c "from legalrag.config import load_settings; s = load_settings(); print('cle lue :', bool(s.anthropic_api_key)); print('modele  :', s.anthropic_model)"
```

Attendu :

```
cle lue : True
modele  : claude-sonnet-5
```

### 5.2 Le modèle est-il proposé ?

```bash
uv run legalrag serve
```

Puis, dans un autre terminal :

```bash
curl -s http://127.0.0.1:8000/api/health
```

Le champ `modeles` doit contenir deux entrées, `groq` et `claude`. Dans la
page, un menu **« Modèle »** apparaît à côté du filtre par document.

### 5.3 Une vraie réponse

```bash
uv run legalrag ask "Quelle autorité délivre les licences de pêche ?" --modele claude
```

---

## Coût

Claude Sonnet 5 est facturé **2 $ par million de tokens en entrée** et
**10 $ par million en sortie** (tarif public, à vérifier sur
<https://www.anthropic.com/pricing>).

Une question de ce chatbot consomme environ :

| Poste | Tokens | Coût |
|---|---|---|
| Entrée (consigne + 5 extraits) | ~3 000 | 0,006 $ |
| Sortie (raisonnement + réponse) | ~1 200 | 0,012 $ |
| **Total par question** | | **≈ 0,018 $** |

Soit environ **55 questions par dollar**, ou **275 questions pour les 5 $**
d'achat minimum. Le raisonnement adaptatif est compté comme de la sortie :
c'est lui qui domine le coût, et c'est pourquoi `ANTHROPIC_EFFORT` vaut `low`
par défaut.

> Ces chiffres sont un ordre de grandeur calculé sur la taille typique d'une
> requête du projet, pas une mesure. Le vrai compteur est dans la console,
> sous **Settings → Usage**.

---

## Diagnostic

### `401 authentication_error`

```
Authentification Anthropic refusee. Verifiez ANTHROPIC_API_KEY.
```

La clé est absente, tronquée, entourée de guillemets, ou a été révoquée.
Vérifier avec la commande de l'[étape 5.1](#51-la-configuration-est-elle-lue).

### `400 invalid_request_error` mentionnant `credit balance`

Le compte n'a pas de crédits. Retour à l'[étape 2](#étape-2--acheter-des-crédits).
La clé est valide : c'est la facturation qui manque.

### Claude n'apparaît pas dans le menu de la page

Le serveur lit `.env` **au démarrage**. Après avoir collé la clé, il faut
l'arrêter (`Ctrl+C`) et le relancer. Si le menu reste absent, vérifier que la
ligne n'est pas commentée par un `#` en début de ligne.

### `400` mentionnant `temperature`

Ne devrait pas se produire : le code n'envoie aucune température à Claude.
Si cela arrive, c'est qu'une modification a réintroduit le paramètre. Sonnet 5
rejette **toute** valeur autre que la sienne par défaut — `0` comme `0.3`.

### La réponse s'arrête au milieu d'une phrase

`ANTHROPIC_MAX_TOKENS` borne le raisonnement **plus** la réponse. L'interface
affiche alors « Réponse interrompue par la limite de longueur ». Monter la
valeur à 8192 dans `.env`, ou baisser `ANTHROPIC_EFFORT`.

### `429 rate_limit_error`

Les plafonds dépendent du palier d'usage du compte, qui monte automatiquement
avec la dépense cumulée. Le SDK réessaie seul jusqu'à quatre fois avec un
repli exponentiel ; si l'erreur persiste, espacer les questions.

---

## Sécurité

- **Ne jamais committer la clé.** `.env` est ignoré par git ; `.env.example`
  ne doit contenir que des valeurs vides.
- **Ne jamais la coller dans un ticket, une capture d'écran ou un message.**
- **Une clé par usage.** Une clé pour le poste de développement, une autre
  pour le déploiement : révoquer l'une n'interrompt pas l'autre.
- **L'image Docker n'embarque aucun secret.** La clé est fournie à
  l'exécution, par `--env-file` ou `-e`.

### Si une clé a fuité

1. Console → **Settings → API keys**
2. Supprimer la clé compromise (**Delete**) — l'effet est immédiat
3. En créer une nouvelle et mettre `.env` à jour
4. Vérifier **Settings → Usage** pour repérer une consommation anormale

---

## Voir aussi

- [`configuration-voyage.md`](configuration-voyage.md) — clé d'embeddings
- [`configuration-qdrant.md`](configuration-qdrant.md) — base vectorielle
- [`configuration-google-drive.md`](configuration-google-drive.md) — accès au corpus
- Section *Deux modèles rédacteurs* du [README](../README.md)
