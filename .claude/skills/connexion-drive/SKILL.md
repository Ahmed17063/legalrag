---
name: connexion-drive
description: Connecter le pipeline à un dossier Google Drive via un compte de service, et diagnostiquer les problèmes d'accès. À utiliser quand l'utilisateur veut configurer l'accès Drive, change de dossier source, obtient « 0 fichier trouvé », une erreur 404, 403, ou un problème de credentials Google.
---

# Connexion à Google Drive

Le pipeline lit le corpus avec un **compte de service** en lecture seule.
Pas d'OAuth interactif : le job tourne sans navigateur ni token à rafraîchir.

## Pourquoi un compte de service et pas OAuth

Dans l'assistant Google Cloud, à la question « À quelles données allez-vous
accéder ? », répondre **« Données de l'application »**, pas « Données
utilisateur ».

Le libellé est trompeur — les fichiers appartiennent bien au Drive
personnel de l'utilisateur. Mais ce que Google demande, c'est *au nom de
qui* le code s'authentifie :

| | |
|---|---|
| Données utilisateur | Client OAuth : navigateur, consentement, jeton qui expire. Inadapté à un job batch |
| Données de l'application | Compte de service : identité propre, clé JSON permanente, aucune interaction |

Le compte de service n'a accès à rien par défaut. C'est **le partage du
dossier avec son adresse** qui lui ouvre l'accès — et qui le rend
révocable en un clic.

## Procédure

1. **Projet Google Cloud** — [console.cloud.google.com](https://console.cloud.google.com/),
   créer un projet ou en réutiliser un.

2. **Activer l'API Drive** — *APIs & Services* → *Library* → « Google Drive
   API » → *Enable*. Sans cette étape, tous les appels échouent en 403.

3. **Créer le compte de service** — *APIs & Services* → *Credentials* →
   *Create credentials* → *Service account*. Un nom suffit. L'étape
   « rôles » se saute : les permissions viennent du partage Drive, pas
   d'IAM.

4. **Générer la clé JSON** — ouvrir le compte créé → onglet *Keys* →
   *Add key* → *Create new key* → **JSON**.

5. **Déplacer le fichier** dans `secrets/` (ignoré par git). Déplacer, pas
   copier : une copie de moins de la clé privée qui traîne dans
   `Downloads`.

   ```bash
   mkdir -p secrets && mv ~/Downloads/<fichier>.json secrets/service-account.json
   ```

6. **Partager le dossier Drive** avec le compte de service. Récupérer son
   adresse :

   ```bash
   python -c "import json;print(json.load(open('secrets/service-account.json'))['client_email'])"
   ```

   Puis sur le dossier Drive : *Partager* → coller l'adresse → **Lecteur**
   → *Envoyer*. Google avertit que ce n'est pas une adresse Gmail : c'est
   normal, continuer.

7. **Renseigner `.env`** :

   ```
   GOOGLE_APPLICATION_CREDENTIALS=./secrets/service-account.json
   DRIVE_FOLDER_ID=<segment après /folders/ dans l'URL du dossier>
   ```

## Vérifier

Ne jamais lire ni afficher le champ `private_key`. Les champs non
sensibles suffisent au diagnostic :

```bash
python -c "
import json
d = json.load(open('secrets/service-account.json'))
print('type        :', d.get('type'))
print('project_id  :', d.get('project_id'))
print('client_email:', d.get('client_email'))
print('cle privee  :', 'presente' if d.get('private_key') else 'ABSENTE')
"
```

Puis vérifier que git ne voit rien :

```bash
git check-ignore -v secrets/service-account.json .env
```

Enfin, tester l'accès réel sans rien facturer :

```bash
python -m legalrag.cli index --dry-run
```

Le nombre de fichiers `.txt` listés doit être non nul.

## Diagnostic

**« 0 fichier(s) .txt trouve(s) »** — c'est le symptôme le plus fréquent, et
le plus trompeur. Le compte de service s'authentifie correctement mais ne
voit rien. **L'étape 6 a été oubliée** : le dossier n'est pas partagé.
L'API Drive ne renvoie pas « accès refusé », elle renvoie une liste vide,
ou un 404 « fichier introuvable ».

**403** — l'API Drive n'est pas activée sur le projet (étape 2), ou le
scope demandé dépasse ce qui a été accordé.

**404 sur un dossier dont l'ID est correct** — même cause qu'un listing
vide : dossier non partagé.

**Fichiers présents mais non listés** — le pipeline ne prend que les `.txt`.
Vérifier `DRIVE_RECURSIVE=true` si le corpus est réparti en sous-dossiers.

**Accents remplacés par des `?` ou `?` dans le terminal** — artefact
d'affichage de la console Windows, pas un problème de données. Pour en
avoir le cœur net, écrire dans un fichier et le relire en UTF-8 plutôt que
de juger sur la sortie console.

## Points d'attention

- Le scope est `drive.readonly` : le pipeline ne peut rien écrire ni
  supprimer, même en cas de bug. Ne pas l'élargir sans raison.
- Ne jamais commiter le JSON. `.gitignore` couvre `secrets/`,
  `*service-account*.json` et `*credentials*.json`.
- Si la clé a fuité, la révoquer dans la console Google : la retirer de
  l'historique git ne suffit pas.
- Les fichiers téléchargés sont mis en cache sous `data/cache/`, indexés
  par identifiant et checksum. Une reprise ne retélécharge pas.
