Date : 15/04/2025
# model-ai
Ce projet est une **application de chatbot intelligent** basée sur le modèle **GPT-4o** (ou un modèle fine-tuné), intégrant un système **RAG (Retrieval-Augmented Generation)** alimenté par du contenu web dynamique issu des sites `bmci.mr` et `masrvi.mr`.

Il permet d'interagir via :

$\bullet$ Une interface web (API REST)  
$\bullet$ Une base de connaissance automatiquement générée à partir des contenus des sites  

Voici les étapes à suivre pour installer le projet :  
## Installation des bibliothèaues python
utilisez la commande suivante pour installer les bibliothèques : 
$$ pip install -r requirements.txt $$

## Configuration 
Créer les variables d'environnement suivantes : 

$\bullet$ QDRANT_URL

$\bullet$ QDRANT_KEY

$\bullet$ OPENAI_API_KEY

## Lancer l'application 
Voici la commande pour lancer le projet : 

uvicorn chatbot_openai:app --reload --port 5002


# Files

## db.py
Ce script Python permet l'enregistrement de messages d'interactions (par exemple entre un utilisateur et un chatbot) dans une base de données MySQL. Il comprend :
$\bullet$ Connexion à une base de données MySQL via le module `mysql.connector`.

$\bullet$ Fonction d’insertion d’un message dans une table `messages` contenant les champs suivants :
  $\quad\rightarrow$ `utilisateur` : identifiant de l'utilisateur  
  $\quad\rightarrow$ `message` : message utilisateur  
  $\quad\rightarrow$ `response` : réponse générée  
  $\quad\rightarrow$ `canal` : type de canal (web, mobile, etc.)

$\bullet$ Gestion des erreurs lors de la connexion ou de l'exécution de requêtes.

### Dépendances
$\bullet$ `mysql-connector-python` : pour la connexion MySQL  
$\bullet$ `pandas` : importé mais **non utilisé** dans ce script (peut être supprimé)

## Configuration attendue

$\bullet$ Base de données : `chatbot`  
$\bullet$ Table : `messages(utilisateur, message, response, canal)`  
$\bullet$ Identifiants de connexion (par défaut dans le script) :
  $\quad\rightarrow$ hôte   
  $\quad\rightarrow$ utilisateur 
  $\quad\rightarrow$ mot de passe 



## chunk.py
Ce script Python permet de **crawling** (explorer) le contenu d’un site web de manière récursive, puis de découper le texte récupéré en **morceaux de taille fixe** pour un traitement ultérieur, par exemple pour de l’analyse, de l’indexation ou une application NLP.

### Fonctionnalités principales

$\bullet$ Récupération récursive du contenu textuel à partir d’une URL de base (avec une profondeur limitée).  
$\bullet$ Utilisation de `requests` pour récupérer les pages web.  
$\bullet$ Utilisation de `BeautifulSoup` pour extraire le contenu textuel HTML.  
$\bullet$ Génération de **blocs de texte de taille fixe** (200 caractères par défaut) pour un traitement par morceaux.  
$\bullet$ Possibilité de paralléliser le crawling avec `ThreadPoolExecutor` (structure en place mais non utilisée dans ce script).

### Comportement du script

$\bullet$ Le script commence à crawler à partir de l’URL : `https://bmci.mr`  
$\bullet$ Il récupère **tout le texte visible** de la page HTML.  
$\bullet$ Il divise le texte obtenu en blocs de 200 caractères via la fonction `process_in_chunks`.

### Dépendances

$\bullet$ `requests` : pour les requêtes HTTP  
$\bullet$ `beautifulsoup4` : pour le parsing HTML  
$\bullet$ `urllib.parse` : pour la gestion des URL  
$\bullet$ `concurrent.futures` : pour une future amélioration de la parallélisation (actuellement non utilisée)


## rag.py
Ce script permet de crawler des sites web, d’en extraire le contenu textuel, de le découper, de le vectoriser avec un modèle d’**embeddings OpenAI**, puis de l’indexer dans une base vectorielle **Qdrant** pour des usages ultérieurs en recherche sémantique ou chatbot.

### Fonctionnalités principales

$\bullet$ Chargement des variables d’environnement via `dotenv` (`.env`) pour sécuriser les clés API.  
$\bullet$ Web scraping de contenu à partir des sites `https://bmci.mr` et `https://masrvi.mr`.  
$\bullet$ Découpage du contenu en morceaux de 300 caractères via `process_in_chunks`.  
$\bullet$ Utilisation d’un modèle d’embedding OpenAI (`text-embedding-3-small`) pour transformer les textes en vecteurs.  
$\bullet$ Indexation dans une collection nommée `"chatbot"` sur un serveur Qdrant auto-hébergé (`localhost:6333`).

### Détails techniques

$\bullet$ Utilise une classe personnalisée `CustomOpenAIEmbedding` pour appeler manuellement l'API OpenAI et créer les embeddings.  
$\bullet$ Le contenu web est encapsulé dans un `DataFrame` Pandas puis transformé en documents via `DataFrameLoader` (LangChain).  
$\bullet$ Indexation initiale dans Qdrant via `Qdrant.from_documents(..., force_recreate=True)` suivie d’une insertion explicite de chaque chunk (`add_texts`).  
$\bullet$ Chronométrage du processus de scraping pour évaluer les performances.

### Dépendances

$\bullet$ `openai`, `langchain`, `langchain_community`, `pandas`, `bs4`, `requests`  
$\bullet$ Serveur **Qdrant** accessible via HTTP (auto-hébergé ou cloud)  
$\bullet$ Fichier `.env` contenant au minimum :
```env
OPENAI_API_KEY=sk-...
QDRANT_KEY=your_qdrant_api_key
```

## chatbot_openai.py
Ce script implémente une API REST avec **FastAPI** pour interagir avec un chatbot basé sur un modèle **GPT-4o** (ou un modèle fine-tuné) via des requêtes HTTP classiques . Il intègre aussi un système de journalisation des conversations dans une base de données MySQL.

## Fonctionnalités principales

$\bullet$ Fournit une route `/chat` pour interagir avec le bot via une interface web ou application.  
$\bullet$ Utilise une session mémoire pour conserver l’historique du dialogue.  
$\bullet$ Intègre des **règles métier strictes** pour que le bot ne réponde qu’aux questions concernant **BMCI** ou **Masrvi**.  
$\bullet$ Journalise chaque interaction dans la table `messages` (utilisation de la fonction `insert_message`).  
$\bullet$ Intègre CORS pour permettre les appels cross-domain, utile pour développement front-end.  

## Détails techniques

$\bullet$ Modèle de base : `ft:gpt-4o-mini-2024-07-18:personal::Arl3TFMr` (fine-tuné).  
$\bullet$ Utilisation de Pydantic pour la validation des requêtes (`UserMessage`).  
$\bullet$ Middleware CORS configuré en mode permissif pour les appels frontend durant le développement.  
$\bullet$ Session définie dans un objet Python (`session`) : remplaçable par Redis ou autre solution persistante si besoin.

## Exemple de logique métier du bot

$\bullet$ Le système est initialisé avec un `SystemMessage` contenant une **personnalité très cadrée** :  
→ Il ne répond qu’aux requêtes liées à BMCI/Masrvi, en français, anglais ou arabe.  
→ Il ne parle **jamais d'autres sujets**, ne fait pas de salutations non sollicitées.  
→ Réponses strictes entre 200 et 500 caractères.

## Endpoints

$\bullet$ `POST /chat` :  
→ Input : `{"message": "Bonjour"}`  
→ Output : `{"response": "...", "history": [...], "intention": "...", "title": "..."}`  


## Dépendances

$\bullet$ `fastapi`, `pydantic`, `twilio`, `langchain`, `openai`, `uvicorn`  
$\bullet$ Modules internes : `custom_prompt` (génération du prompt), `insert_message` (MySQL), etc.
