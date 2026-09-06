# Configurer l'accès Google Drive

Guide pas à pas pour donner au pipeline un accès en lecture au dossier Drive
qui contient le corpus.

**Durée : environ 10 minutes.** Aucun coût, aucune carte bancaire.

---

## Ce qu'on va faire, et pourquoi

Le pipeline doit lire des fichiers sur Drive sans qu'un humain soit devant
l'écran. On lui donne donc une **identité propre** — un *compte de service* —
plutôt que de lui faire emprunter la vôtre.

```
┌──────────────────┐        ┌─────────────────────┐        ┌──────────────┐
│  Google Cloud    │        │  Compte de service  │        │ Dossier      │
│  (projet)        │───────▶│  legal-ai@....com   │───────▶│ Drive        │
│                  │  crée  │  + clé JSON         │ partagé│ (corpus)     │
└──────────────────┘        └─────────────────────┘  avec  └──────────────┘
                                      │
                                      ▼
                             secrets/service-account.json
```

Deux idées à retenir :

1. **Le compte de service est un utilisateur à part entière**, avec sa propre
   adresse e-mail. Il n'hérite d'aucun de vos droits.
2. **Il n'accède à rien tant que vous ne partagez pas explicitement** le
   dossier avec son adresse. C'est ce qui rend l'accès révocable en un clic,
   et c'est aussi l'étape qu'on oublie le plus souvent (voir
   [Diagnostic](#diagnostic)).

---

## Étape 1 — Créer ou choisir un projet Google Cloud

Aller sur [console.cloud.google.com](https://console.cloud.google.com/).

En haut de la page, le sélecteur de projet. Créer un nouveau projet
(*New Project*) ou en réutiliser un existant. Un nom suffit, par exemple
`legal-ai`.

> Un projet Google Cloud est gratuit tant qu'on n'active pas de service
> facturé. L'API Drive en lecture ne l'est pas.

---

## Étape 2 — Activer l'API Google Drive

**Cette étape est obligatoire.** Sans elle, tous les appels échouent en 403,
même avec des credentials parfaitement valides.

1. Menu ☰ → **APIs & Services** → **Library**
2. Chercher « Google Drive API »
3. Cliquer dessus, puis **Enable**

---

## Étape 3 — Créer le compte de service

1. Menu ☰ → **APIs & Services** → **Credentials**
2. **+ Create credentials** → **Service account**

Un assistant s'ouvre et pose une question qui prête à confusion :

> **À quelles données allez-vous accéder ?**
> ◯ Données utilisateur
> ◯ Données de l'application

**Choisir « Données de l'application ».**

Le libellé induit en erreur : vos fichiers sont bien dans *votre* Drive
personnel, donc l'instinct pousse vers « Données utilisateur ». Mais la
question porte en réalité sur **l'identité qui s'authentifie** :

| Option | Ce que ça crée | Conséquence |
|---|---|---|
| Données utilisateur | Un client **OAuth** | Le programme ouvre un navigateur, vous cliquez « Autoriser », il reçoit un jeton qui expire et doit être rafraîchi. Inutilisable pour un job automatique |
| **Données de l'application** | Un **compte de service** | Le programme a sa propre identité et une clé permanente. Aucune interaction humaine |

Ensuite :

3. **Nom du compte de service** : quelque chose de parlant, par exemple
   `legalrag-indexeur`. Google en déduit une adresse e-mail.
4. **Étape « Rôles » (permissions du projet) : la passer.** Cliquer
   directement sur *Continue* puis *Done*. Les droits ne viennent pas d'IAM
   ici, mais du partage Drive à l'étape 6.

---

## Étape 4 — Générer la clé JSON

1. Dans la liste **Credentials**, cliquer sur le compte de service que vous
   venez de créer
2. Onglet **Keys** (Clés)
3. **Add key** → **Create new key**
4. Choisir le format **JSON**
5. **Create** — le fichier se télécharge automatiquement

Le fichier ressemble à `legal-ai-507810-b723e67838ba.json` et contient :

```json
{
  "type": "service_account",
  "project_id": "votre-projet",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...",
  "client_email": "legalrag-indexeur@votre-projet.iam.gserviceaccount.com",
  ...
}
```

> ⚠️ **Ce fichier est un secret.** Le champ `private_key` donne un accès
> complet au compte de service. Ne jamais le commiter, l'envoyer par
> messagerie ou le coller dans un ticket.
>
> Google ne conserve pas de copie de la clé privée. Si vous la perdez, il
> faut en générer une nouvelle.

---

## Étape 5 — Ranger le fichier

**Déplacer** le fichier — ne pas le copier. Une copie de moins qui traîne
dans `Downloads`, c'est une fuite potentielle de moins.

```bash
mkdir -p secrets
mv ~/Downloads/<votre-fichier>.json secrets/service-account.json
```

Sous Windows / PowerShell :

```powershell
mkdir secrets -Force
move $HOME\Downloads\<votre-fichier>.json secrets\service-account.json
```

Vérifier que git l'ignore bien :

```bash
git check-ignore -v secrets/service-account.json
```

La commande doit répondre en citant la règle du `.gitignore`. Si elle ne
renvoie **rien**, le fichier n'est pas ignoré : ne surtout pas commiter, et
corriger le `.gitignore` d'abord.

---

## Étape 6 — Partager le dossier Drive

**C'est l'étape décisive, et celle qu'on oublie.**

Jusqu'ici le compte de service existe, mais il ne voit rien de votre Drive.

### 6.1 Récupérer son adresse

```bash
python -c "import json; print(json.load(open('secrets/service-account.json'))['client_email'])"
```

Vous obtenez une adresse de la forme :

```
legalrag-indexeur@votre-projet.iam.gserviceaccount.com
```

### 6.2 Partager

1. Ouvrir le dossier Drive contenant le corpus
2. Clic droit → **Partager**
3. Coller l'adresse du compte de service
4. Choisir le rôle **Lecteur** (pas Éditeur : le pipeline n'écrit rien)
5. **Envoyer**

> Google affiche un avertissement indiquant que ce n'est pas une adresse
> Gmail habituelle, ou que le destinataire ne pourra pas être notifié.
> C'est normal, continuer.

### 6.3 Récupérer l'ID du dossier

Dans l'URL du dossier :

```
https://drive.google.com/drive/folders/1TwWEQk33CI9M8v_mdj2gpKPoOKariJm2
                                        └──────── l'ID est ce segment ────────┘
```

---

## Étape 7 — Renseigner la configuration

Dans le fichier `.env` à la racine du dépôt (créé depuis `.env.example`) :

```
GOOGLE_APPLICATION_CREDENTIALS=./secrets/service-account.json
DRIVE_FOLDER_ID=1TwWEQk33CI9M8v_mdj2gpKPoOKariJm2
DRIVE_RECURSIVE=true
```

`DRIVE_RECURSIVE=true` fait explorer aussi les sous-dossiers.

---

## Étape 8 — Vérifier

### 8.1 Le fichier de credentials est lisible et complet

Cette commande n'affiche **aucun secret** — seulement les champs
d'identification :

```bash
python -c "
import json
d = json.load(open('secrets/service-account.json'))
print('type         :', d.get('type'))
print('project_id   :', d.get('project_id'))
print('client_email :', d.get('client_email'))
print('cle privee   :', 'presente' if d.get('private_key') else 'ABSENTE')
"
```

Attendu :

```
type         : service_account
project_id   : votre-projet
client_email : legalrag-indexeur@votre-projet.iam.gserviceaccount.com
cle privee   : presente
```

### 8.2 L'accès au dossier fonctionne

```bash
python -m legalrag.cli index --dry-run
```

Cette commande liste le Drive, analyse les noms de fichiers et estime le
coût, **sans effectuer le moindre appel facturé**.

Attendu :

```
INFO  legalrag.drive     Client Drive initialise (compte de service : legalrag-indexeur@...)
INFO  legalrag.pipeline  530 fichier(s) .txt trouve(s) sur Drive
```

Si le nombre de fichiers est non nul, **la configuration est terminée**.

---

## Diagnostic

### « 0 fichier(s) .txt trouve(s) »

**Cause la plus fréquente : l'étape 6 n'a pas été faite**, ou l'adresse
collée n'était pas la bonne.

Le compte de service s'authentifie correctement — d'où l'absence de message
d'erreur — mais ne voit aucun fichier. L'API Drive ne répond pas « accès
refusé » : elle renvoie une liste vide.

Vérifier dans les paramètres de partage du dossier que l'adresse
`...iam.gserviceaccount.com` y figure bien.

### Erreur 403

L'API Drive n'est pas activée sur le projet → refaire l'étape 2.

### Erreur 404 sur un ID de dossier pourtant correct

Même cause qu'un listing vide : le dossier n'est pas partagé avec le compte
de service.

### « Fichier de credentials Google introuvable »

Le chemin dans `GOOGLE_APPLICATION_CREDENTIALS` ne pointe pas sur un
fichier existant. Il est relatif à la racine du dépôt.

### Des fichiers existent mais ne sont pas listés

Le pipeline ne prend que les `.txt`. Si le corpus est réparti en
sous-dossiers, vérifier `DRIVE_RECURSIVE=true`.

### Les accents s'affichent en `?` ou `�`

Artefact d'affichage de la console Windows, **pas** un problème de données.
Pour en avoir le cœur net, écrire la sortie dans un fichier et le relire en
UTF-8 plutôt que de juger sur l'affichage du terminal.

---

## Sécurité

| Règle | Pourquoi |
|---|---|
| Le scope demandé est `drive.readonly` | Le pipeline ne peut rien écrire ni supprimer sur votre Drive, même en cas de bug |
| Le rôle partagé est **Lecteur** | Même raison, côté Drive |
| Le JSON ne quitte jamais `secrets/` | Ce dossier est dans `.gitignore` |
| Aucune clé en dur dans le code | Tout passe par des variables d'environnement |

### Révoquer l'accès

Deux niveaux, du plus rapide au plus radical :

1. **Retirer le partage Drive** — le compte de service ne voit plus rien,
   immédiatement. Suffit dans la plupart des cas.
2. **Supprimer la clé** dans Google Cloud → *Credentials* → le compte de
   service → onglet *Keys* → supprimer. À faire si le JSON a fuité.

Si le fichier JSON a été commité par erreur, **supprimer la clé dans la
console** : la retirer de l'historique git ne suffit pas, elle a déjà été
exposée.

---

## Voir aussi

- [README](../README.md) — vue d'ensemble du pipeline
- [Skill `connexion-drive`](../.claude/skills/connexion-drive/SKILL.md) — la
  même procédure, sous une forme exploitable par Claude Code
- [Skill `connexion-voyage`](../.claude/skills/connexion-voyage/SKILL.md) —
  l'équivalent pour la clé d'API d'embeddings
