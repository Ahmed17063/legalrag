import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to crawl a website recursively
def crawl_website(base_url, visited=None, depth=2):
    print("crawling web site : " + base_url)
    if visited is None:
        visited = set()
    if depth == 0 or base_url in visited:
        return ""

    visited.add(base_url)
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
    except requests.RequestException as e:
        print(f"Failed to fetch {base_url}: {e}")
        return ""

    # Extract main content from the page
    page_text = soup.get_text(separator="\n", strip=True)
    content = f"Content from {base_url}:\n{page_text}\n\n"
    return content

# Function to process the content in 300-character chunks
def process_in_chunks(content, chunk_size=200):
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i + chunk_size]
        yield chunk  # Yield each chunk as a generator


# Crawl the website and get the full content
#content = crawl_website(base_url="https://bmci.mr")
#content += crawl_website(base_url="https://bmci.mr/ouvrir-votre-compte-en-ligne/")
#content += crawl_website(base_url="https://bmci.mr/app/")
#content += crawl_website(base_url="https://bmci.mr/entreprise?lang=fr")
#content += crawl_website(base_url="https://bmci.mr/produits_islamiques?lang=fr")
#content += crawl_website(base_url="https://bmci.mr/mot-du-dg?lang=fr")
#content += crawl_website(base_url="https://bmci.mr/sponsoring?lang=fr")

text =[
    "Conditions Générales : Aucune information apparaissant sur le présent site ne saurait être",
    "Voici les lient pour telecharger l'application BMCI Mobile : ios : https://apps.apple.com/mr/app/bmci-mobile-mauritanie/id1467188896, android : https://play.google.com/store/apps/details?id=mr.bmcimobile.www&pcampaignid=web_share",
    "Voici les lient pour telecharger l'application Masrvi : ios : https://apps.apple.com/mr/app/masrvi/id1540766096, android : https://play.google.com/store/apps/details?id=com.tagpay.mwallet.masrvi&pcampaignid=web_share",
    "La BMCI propose plusieurs types de cautions et avals : Caution de Soumission, Caution de Bonne Exécution, Caution d’Avance de démarrage, Caution Retenue de Garantie, Caution Douanière, Autres Cautions, et Traite Avalisée.",
    "Le crédit immobilier pour l'acquisition à la BMCI a un plafond de 2 000 000 MRU, une durée maximale de 15 ans, un taux de 11%, et une commission de 1.5%.",
    "La BMCI propose plusieurs types de crédits aux particuliers : Crédit Standard, Silver, Premium, Electroménager, et les crédits pour Avance Fête, Scolaire, Vacance.",
    "La BMCI propose plusieurs types de cartes guichet : Carte Gimtel, Carte Gimtel Prépayée, Carte Visa Electron, Carte Visa Gold, et Carte Visa Prépayée.",
    "Pour un crédit de 100000 MRU sur 10 mois avec un taux d'intérêt de 1%, voici la simulation :\n\n- Montant du crédit : 100000 MRU\n- Durée : 10 mois\n- Taux d'intérêt : 1%\n\nMensualité : 23345 MRU par mois avec un intérêt de 13%.\n\nCommission : 1.5%\n\nTotal à rembourser : 233450 MRU.",
    "La BMCI dispose d’un réseau de 47 agences et 48 GABs répartis sur l’ensemble du territoire mauritanien, voici quelques agences très fréauentées : Agence centrale : A Nouakchott, Capital, Agence Ksar : A Ksar, Agence Toujounin : A Toujounin, Agence Nouadhbou, Agence Zouerrat, Agence Rosso, Agence Nouadhibou 2",
    "considérée comme une offre de services ou de produits, notamment produits bancaires ou d´assurance,",
    "émanant de la Banque Mauritanienne pour le Commerce International, ni comme une offre ou \
    la sollicitation d´une offre d´achat ou de vente de valeurs mobilières ou de tout autre produit d´investissement.",
    "La Banque Mauritanienne pour le Commerce International décline toute responsabilité quant au contenu des présentes",
    "pages et à l´utilisation qui pourrait en être faite par quiconque. La BMCI s’efforce d’assurer l’exactitude",
    "et la mise à jour des informations diffusées dans ce site, dont elle se réserve le droit de corriger",
    "Comment puis-je ouvrir un compte en ligne? Rien de plus simple. Rendez-vous sur la page « Devenir client » sur notre site.",
    "Suivez les instructions. Vous pouvez également contacter notre équipe de conseiller au siège de la BMCI.",
    "Notre service client est à votre disposition 0022247298000 24/7 BANQUE MAURITANIENNE POUR LE COMMERCE INTERNATIONAL Société anonyme au capital de 1 Milliard MRU Avenue de Gamal Abdel Nasser, BP: 622. Nouakchott, Maurtitanie",
    "MASRVI permet aux entreprises de gérer à distance et de manière électronique le paiement des salaires de leurs employés, offrant ainsi un gain de temps et une gestion simplifiée.",
    "Le crédit immobilier pour la construction chez la BMCI a un plafond de 3 000 000 MRU, une durée maximale de 15 ans, un taux de 11%, et une commission de 1.5%.",
    "Le DAT (Dépôt à Terme) est un produit d'épargne à durée déterminée proposé par la BMCI",
    "La BMCI propose plusieurs types de crédits aux particuliers : Crédit Standard, Silver, Premium, Electroménager, et les crédits pour Avance Fête, Scolaire, Vacance."
    "Pour ouvrir un compte MASRVI, téléchargez l'application depuis le Play Store ou l'App Store, inscrivez votre numéro de téléphone et votre NNI, puis identifiez-vous dans une agence BMCI ou MASRVI pour activer votre compte."
]
# Process the content in 300-character chunks
#for chunk in process_in_chunks(content):
#    text.append(chunk)
