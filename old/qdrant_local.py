import random
import time
from typing import Any
import uuid
import pandas as pd
from langchain_community.vectorstores import Qdrant
from qdrant_client import models, qdrant_client

from utils import *
from langchain_core.documents import Document


# Optional: delete and recreate the collection
def recreate_collection(collection_name):
    print(f"recreating collection {collection_name}")
    collection =  client_qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE,
            on_disk=True,  # Reduce RAM usage during bulk uploads by storing vectors directly on disk
        ),
        # Reduce CPU/memory pressure during bulk operations
        hnsw_config=models.HnswConfigDiff(m=0), # Disable HNSW graph
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0), # Disable indexing during upload
        shard_number=2 # Enables parallel upload
    )
    print("")
    return collection

# Step 4: Initialize vector store
def store_data_in_collection(collection_name, documents):    

    # 1. Convert to Document
    masrvi1 = Document(
        page_content = "'Transfert INTEROP', 'Depuis BMCI', 'xx MRU (Montant: xx', 'MRU, frais', 'yy MRU, taxes', 'zz', MRU) paye chez Transfert d'argent, '(REF)_', 'OK', 'Vous', 'riuis', '1/,24 IvnUTulbitiiu', 'argent', '-xx MRU Taxe', 'yy MRU', 'xx MRU', 'Terminer'",
        metadata = {
            "type": "transfert",
            "bank": "Masrvi",
            "response": "Il s'ajit d'une capture d'ecran de l'application Masrvi, c'est un transfert via Masrvi (BMCI) d'un montant de xx MRU, frais de yy MRU et taxe de zz MRU effectué le dd mm. L'opération a été éffectuée et aucun problème détecté."
        }
    )

    masrvi2 = Document(
        page_content="'Paiement facture', 'depuis compte', 'xx MRU (Montant: xx MRU;', 'frais', 'yy MRU, taxes', 'yy MRU)', 'paye chez Entity', 'du', 'Tresor (REF_Entiry)_', 'OK', 'Vous', 'DIIEClIUI', 'Otiitiuit', 'uu !esor', '-20,00 MRU', '20,00 MRU'",
        metadata= {
            "type": "Paiement facture",
            "bank": "Masrvi",
            "response": "Il s'ajit d'une capture d'ecran de l'application Masrvi,c'est un paiement de facture depuis Masrvi, la facture de Entity avec un montant xx MRU. L'opération a été éffectuée et aucun pproblème détecté.",
            "category" : "Support Masrvi"  
      }
    )
    masrvi3 = Document(
        page_content="'FirstName LastName', 'xxx MRU', '+222 phoneNumber', 'Transfert d'argent', 'Paiement Commercant', 'Paiement Factures', 'Recharge telephonique', 'Recharge compte'', 'Autres', 'Agences', 'Releves de', '@B', 'Masrvi', 'B', 'compte', '00', 'Accueil', 'Produits', 'Menu'",
        metadata= {
            "type": "Page accueil Masrvi",
            "bank": "Masrvi",
            "response": "Il s'ajit d'une capture d'ecran de l'application Masrvi, c'est la page d'acceuil de Masrvi du FirstName LastName, depuis cette page vous pouvez faire toutes les operations Masrvi (Trasfert d'argent, Paiement commerçant, paiement facture, recharge telephonique, recharger votre compte Masrvi depuis votre compte BMCI ou votre carte, faire un retrait depuis un guichet BMCI ....)",
            "category" : "Support Masrvi"  
      }
    )

    masrvi3_1 = Document(
        page_content="'hh: mm', 'mm', 'FirstName LastName, 'xxx MRU', '+222 phoneNumber', 'Transfert', 'Paiement', 'dargent', 'Commerçant', 'Paiement', 'Recharge', 'E', 'Factures', '2', 'téléphonique', 'Recharge', 'Retrait', 'compte', 'dargent', 'Autres', 'Agences', 'Relevés de', '[8', 'Masrvi', 'compte', '00', 'Accueil', 'Produits', 'Menu'",
        metadata= {
            "type": "Page accueil Masrvi",
            "bank": "Masrvi",
            "response": "Il s'ajit d'une capture d'ecran de l'application Masrvi, c''est la page d'acceuil de Masrvi du client FirstName LastName, depuis cette page vous pouvez faire toutes les operations Masrvi (Trasfert d'argent, Paiement commerçant, paiement facture, recharge telephonique, recharger votre compte Masrvi depuis votre compte BMCI ou votre carte, faire un retrait depuis un guichet BMCI ....)",
            "category" : "Support Masrvi"  
      }
    )


    masrvi4 = Document(
        page_content="'Transferts', 'Transfert', 'GIMTEL' ",
        metadata= {
            "type": "Faire un trasfert Masrvi",
            "bank": "Masrvi",
            "response": "Il s'ajit d'une capture d'ecran de l'application Masrvi, c'est la page pour faire un virement Masrvi, vous pouvez faire un trasfert vers un compte Masrvi en cliquant sur Treansfert ou vers unu autre wallet en cliquant sur Gimtel",
            "category" : "Support Masrvi"  
      }
    )

    masrvi5 = Document(
        page_content="'Transfert', 'Telephone', 'Banque', 'Vers', 'Mauritanie', '+222', 'Numero de telephone', 'Montant', 'MRU', 'Description', 'Mes contacts', 'Rechercher par nom',  '+222 phoneNumber1', '+222 phoneNumber2', '+222 phoneNumber3'",
        metadata= {
            "type": "Faire un trasfert Masrvi Masrvi",
            "bank": "Masrvi",
            "response": "Il s'ajit d'une capture d'ecran de l'application Masrvi, cette page est accessible depuis la page d'accueil -> Transfer. Elle est dediée faire un virement Masrvi vers un compte Masrvi, vous devez saisir le numero de telephone du destinataire ou rechercher dans la liste de vos contacts et saisir le montant. la description est facultative",
            "category" : "Support Masrvi"  
      }
    )

    masrvi5_1 = Document(
        page_content="'Transfert', 'Telephone', 'Banque', 'Vers', 'Mauritanie', '+222', 'Numero de telephone', 'Montant', 'MRU', 'Description', 'Mes contacts', 'Rechercher par nom', '+222 phoneNumber1', '+222 phoneNumber2', '+222 phoneNumber3'",
        metadata= {
            "type": "Faire un trasfert Masrvi Masrvi",
            "bank": "Masrvi",
            "response": "Il s'ajit d'une capture d'ecran de l'application Masrvi, cette page est accessible depuis la page d'accueil -> Transfer. Elle est dediée faire un virement Masrvi vers un compte Masrvi, vous devez saisir le numero de telephone du destinataire ou rechercher dans la liste de vos contacts et saisir le montant. la description est facultative",
            "category" : "Support Masrvi"  
      }
    )


    masrvi6 = Document(
        page_content="'Transferts', 'GIMTEL', 'Numero destinataire', 'xxxxxxxx', 'Montant', 'MRU', 'Suivant'",
        metadata= {
            "type": "Faire un trasfert Gimtel",
            "bank": "Masrvi",
            "response": "Il s'ajit d'une capture d'ecran de l'application Masrvi, cette page est accessible depuis la page d'accueil -> Transfer. Pour faire un virement GIMTEL (Vers un compte non Masrvi), Ouvrir l'application Masrvi, connectez-vous, puis choisir 'Transfert d'argent' puis GIMTEL. Elle est dediée pour faire un virement Gimtel (Masrvi vers une autre wallet), veuillez saisir le numero de telephone du destinataire et le montant et cliquer sur suivant, vous allez apres choisir la wallet et confirmer pour faire le virement",
            "category" : "Support Masrvi"  
      }
    )

    masrvi7 = Document(
        page_content="'Paiement Factures', 'Facturiers', 'Administration & impots', 'Rechercher par nom ou code', 'SNDE', 'SOMELEC', 'SOMELEC MOUNSIF', 'Moov', 'Mauritel (Internet)', 'KDn', 'M2E', 'MZE', 'RIMATEL', 'SNDE'",
        metadata= {
            "type": "Payer une facture",
            "bank": "Masrvi",
            "response": "Il s'ajit d'une capture d'ecran de l'application Masrvi, Pour payer une facture ou impôt depuis Masrvi, ouvrir votre application et cliquer sur Paiement facture. Paiement facture. Elle est dediée pour payer une facture (SNDE, SOMELEC, WIFI, Assurance, ....). Pour payer une facture, veuillez choisir le service, saisir votre reference et payer le montant du ou saisir. Vous pouvez aussi cliquer sur 'Administration & impots' pour payer vos factures administratives comme la vignette de votre voiture ou obtenir des documents officiels...",
            "category" : "Support Masrvi"  
      }
    )


    masrvi7_1 = Document(
        page_content="'hh:mm', 'mm', 'Paiement Factures', 'Facturiers', 'Administration & impôts', 'Rechercher par nom ou code', 'SNDE', 'SNDE', 'SOMELEC', 'Sumlll(', 'SOMELEC MOUNSIF', 'SoMff', 'Moov', 'Mauritel (Internet)', 'Mouuile', 'MZE', 'MZE', 'RIMAAIEL', ''",
        metadata= {
            "type": "Payer une facture",
            "bank": "Masrvi",
            "response": "Il s'ajit d'une capture d'ecran de l'application Masrvi, Pour payer une facture ou impôt depuis Masrvi, ouvrir votre application et cliquer sur Paiement facture. Paiement facture. Elle est dediée pour payer une facture (SNDE, SOMELEC, WIFI, Assurance, ....). Pour payer une facture, veuillez choisir le service, saisir votre reference et payer le montant du ou saisir. Vous pouvez aussi cliquer sur 'Administration & impots' pour payer vos factures administratives comme la vignette de votre voiture ou obtenir des documents officiels...",
            "category" : "Support Masrvi"  
      }
    )


    masrvi8 = Document(
        page_content="'hh:mm 0', 'Paiement Factures', 'Facturiers', 'Administration & impôts', 'Rechercher par nom ou code', 'u', 'Direction Générale du Trésor', 'DGI-Vignette', 'Tribunal de commerce', 'AL Ibdaa Bank'",
        metadata= {
            "type": "Payer une facture Administrative",
            "bank": "Masrvi",
            "response": "Il s'ajit d'une capture d'ecran de l'application Masrvi, Pour payer une facture administrative ou impôt depuis Masrvi, ouvrir votre application et cliquer sur Paiement facture. Elle est dediée pour faire payer vos factures administratives comme la vignette de votre voiture ou les frais des documents officiels comme le passport, carte d'identite, carte grise, extrait d'acte de naissance, timbre, pour ce faire cliquer sur 'Direction Général du Trésor', choisir le service, saisir les informations demandées et payer. Depuis cette page vous pouvez aussi payer vos amendes routieres  en cliquant sur 'Direction Général du Trésor' choisir 'Amende routiere', saisir la reference de l'amende et payer le montant afficher.",
            "category" : "Support Masrvi"  
      }
    )

    masrvi9 = Document(
        page_content="'Menu', 'Transferts en attente', 'Code confirmation', 'Changer mon code secret', 'Nous', 'appeler', 'Nous ecrire', 'En savoir plus', 'Preferences', 'Securite', 'Deconnexion', '88', 'Accueil', 'Produits', 'Menu'",
        metadata= {
            "type": "Parametres",
            "bank": "Masrvi",
            "response": "Il s'ajit d'une capture d'ecran de l'application Masrvi, Pour modifier les paraetres de votre compte Masrvi ou application , ouvrir votre application et cliquer sur Menu (en bas à droite). cette page est dediée modifier les parametres de votre compte et Application Masrvi, cliquer sur 'Code confirmation' pour configurer le canal de communication pour recevoir les code de connexion a votre compte Masrvi, pour pouvez choisir SMS ou E-mail. Depuis cette page vous pouvez egalement changer votre code secret, appelez ou envoyer un email a Masrvi, vous pouvez egalement changer les preferences de votre application et la securite de votre compte",
            "category" : "Support Masrvi"  
      }
    )

    bmci1 = Document(
        page_content="'hh:mm', '73', 'Comptes', 'Mes crédits', 'Solde disponible:', 'MRU', 'COMPTES CHEQUES PERSONNELS', 'xxxx-xxxx-xx', 'MES MOUVEMENTS', 'PAIMENT', 'xxxx MRU', 'dd/mm/yyyy', 'RET GAB', '396157', 'xxxx MRU', 'dd/mm/yyyy', 'PAIMENT VISA', '392404', 'xxxx MRU', 'dd/mm/yyyy','PAIMENT VISA', '392404', 'xxxx MRU', 'dd/mm/yyyy', 'VRT MASRVI WZB', 'xxxx MRU', 'dd/mm/yyyy', 'PAIMENT VISA', '384549', 'xxxx MRU', 'dd/mm/yyyy', 'RET GAB', '389936', 'xxxx MRU', 'dd/mm/yyyy', 'PAIMENT VISA', '384549', 'xxxx MRU', 'dd/mm/yyyy', 'Accueil', 'Crédit', 'Demandes', 'Services', 'Autres'",
        metadata= {
            "type": "Acceuil",
            "bank": "BMCI Mobile",
            "response": "Il s'ajit d'une capture d'ecran de la page d'accueil de l'application Banque en ligne BMCI Mobile, elle presente la sole disponible du client, la liste des comptes du client et les mouvements recents effectues sur les comptes, cliquer sur 'Mes crédits' pour voir la liste des credits (Prets) en cours du clients, clique sur la fleche à droite des mouvements pour voir les mouvements effectues pendant une periode donnees. En bas de la page vous avez 'Crédits' pour demander, renouveller ou consulter un credit. Vous avez egalement 'Demandes' pour envoyer une demande a la banque, 'Services' pour demander un chéquier, une carte, payer une facture, faire une virement ou demander des documents. En bas à droite de la page 'Autres' pour modifier les parametres de votre compte ou contacter la banque.",
            "category" : "Support BEL"  
      }
    )

    bmci2 = Document(
        page_content="'mm:hh', 'DEMANDE DE CRÉDIT', 'Crédit instantané', 'Nouveau crédit avec', 'validation', 'Renouvellement de crédit', 'Historique des crédits', 'Accueil', 'Crédit', 'Demandes', 'Services', 'Autres",
        metadata= {
            "type": "Crédit (Prets)",
            "bank": "BMCI Mobile",
            "response": "Il s'ajit d'une capture d'ecran de la page dedier pour les crédits de l'application Banque en ligne BMCI Mobile, elle est accessible depuis la page d'acceuil -> Crédit (en bas de l'ecran), elle permet de demander un credit (pret) instantanne, demander ou obtenir un credit (pret) avec validation, renouveller un credit (pret) et consulter l'historique des credits (prets). Pour obtenir ou demander un pret, Ouvrir l'application BMCI Mobie, connecter vous et choisissez Crédit (en bas de l'ecran) et suivre les instructions.",
            "category" : "Support BEL"  
      }
    )

    bmci3 = Document(
        page_content="'mm:hh', 'MESSAGES', 'Reçus', 'Envoyés', 'Aucun message trouvé', 'Accueil', 'Crédit', 'Demandes', 'Services', 'Autres'",
        metadata= {
            "type": "Demandes",
            "bank": "BMCI Mobile",
            "response": "Il s'ajit d'une capture d'ecran de la page dedier pour les demandes de l'application Banque en ligne BMCI Mobile, elle est accessible depuis la page d'acceuil -> Demande (en bas au milieu de l'ecran). Pour faire une demande, cliquer sur '+', choisir l'objet de la demande (Attestation d'autofinancement, attestation d'engagement, attestation de non engagement, attestation de prise en charge, attestation de solde, attestation de trasfert, Carte capturée GAB, demande de mise à disposition d'une carte,prise de rendez-vous, reclamations .....), et redigez votre message, vous pouvez egalement joindre des documents",
            "category" : "Support BEL"  
      }
    )

    bmci4 = Document(
        page_content="'mm:hh', 'SERVICES', 'Chéquier', 'Cartes', 'Paiement de factures', 'Virements', 'Documents client', 'Accueil', 'Crédit', 'Demandes', 'Services', 'Autres'",
        metadata= {
            "type": "Services",
            "bank": "BMCI Mobile",
            "response": "Il s'ajit d'une capture d'ecran de la page dedier pour les services de l'application BMCI Mobile, elle est accessible depuis la page d'acceuil -> Service (en bas de l'ecran). Elle englobe tous les services proposés par la BMCI, a savoir le chéquier, les cartes, paiements des factures, virements, documents clients ....",
            "category" : "Support BEL"  
      }
    )

    bmci5 = Document(
        page_content="'hh:mm', 'VIREMENTS', 'Effectuer un virement', 'Liste des Bénéficiaires', 'Ajouter un bénéficiaire', 'Historique des virements', 'immédiatsldifférés', 'Historique des virements', 'permanents'",
        metadata= {
            "type": "Services - Virements",
            "bank": "BMCI Mobile",
            "response": "Il s'ajit d'une capture d'ecran de la page dedier pour les virements de l'application BMCI Mobile, elle est accessible depuis la page d'acceuil -> Service (en bas de l'ecran) -> Virements. Elle englobe tous les services cencernant les virements proposés par la BMCI, a savoir le 'Effectuer un virement' 'List des bénéficiaires', 'Ajouter un bénéficiaire' et voir l'historique des virements",
            "category" : "Support BEL"  
      }
    )

    bmci6 = Document(
        page_content="'hh:mm', 'GESTION DES CHÉQUIERS', 'Demande de chéquier', 'Demande dopposition', 'sur chèque'",
        metadata= {
            "type": "Services - Chéquier",
            "bank": "BMCI Mobile",
            "response": "Il s'ajit d'une capture d'ecran de la page dedier pour les chequiers de l'application BMCI Mobile, elle est accessible depuis la page d'acceuil -> Service (en bas de l'ecran) -> Chéquier. Pour demander un chequier, ou une opposition sur une cheque, Depuis cette page, accessible depuis la page d'acceuil -> Service (en bas de l'ecran) -> Chéquier, le client peux faire une demande de chéquier, ou demande l'opposition sur une chéque",
            "category" : "Support BEL"  
      }
    )

    bmci7 = Document(
        page_content="'hh:mm', 'PAIEMENT DE FACTURES', 'Favoris', 'Aucun identifiant trouvée', 'Liste des facturiers', 'Somelec', 'SOMELEC', 'SNDE', 'SNDE'",
        metadata= {
            "type": "Services - Factures",
            "bank": "BMCI Mobile",
            "response": "Il s'ajit d'une capture d'ecran de la page dedier pour le paiement des factures depuis l'application BMCI Mobile, elle est accessible depuis la page d'acceuil -> Service (en bas de l'ecran) -> 'Paiement de factures'. Depuis cette page, le client peux voir la liste des facturiers (SOMELEC, SNDE, ...), choisir le facturier, saisir la reference et payer la facture",
            "category" : "Support BEL"  
      }
    )

    bmci8 = Document(
        page_content="'hh:mm', 'CARTES', 'bmsh', 'ëdgJl ëJbw', 'Lzjsll sLyjI', 'Bonque Mouritonienne pour le Commerce Internotional', 'PRODUIT CARTE VISA', 'ELECTRON', 'XXXXXXXXXXXXXXXX', 'EXPIRE', 'a', '1n9i0096', 'quxll +ine', 'Ququllollo lzallo', 'STATUT DE CARTE :', 'Activé', 'MOUVEMENTS DE CARTE :', 'dd/mm/yyyy', 'm1', 'MRU', '07000 RETRAIT (ATM)', '0300000024 GAB2 PARTICULIERS', 'Nouakchott', 'MR', 'dd/mm/yyyy', '05000 PAIEMENT', 'm2 EUR', '4036040 VERIFDIPLOMA', 'VERSAILLES', 'FR', 'dd/mm/yyyy', 'm3', 'MRU', '07000 RETRAIT (ATM)', '0300000024 GAB2 PARTICULIERS', 'Nouakchott', 'MR', 'dd/mm/yyyy', 'm4', 'USD', '05000 PAIEMENT', '463420000557165 NETFLIXCOM', '4087249160', 'NL', 'dd/mm/yyyy'",
        metadata= {
            "type": "Services - Cartes",
            "bank": "BMCI Mobile",
            "response": "Il s'ajit d'une capture d'ecran de la page dedier pour les services des cartes BMCI depuis l'application BMCI Mobile, elle est accessible depuis la page d'acceuil -> Service (en bas de l'ecran) -> Cartes. Elle presente les cartes de l'utilisateur, et les derniers mouvements effectues sur les cartes. Pour faire des operations sur les cartes, cliquer sur les 3 traits en haut à droite, et depuis cette page, le client peux commander une carte, voir l'historique des demandes des cartes ou faire une opposition sur les cartes",
            "category" : "Support BEL"  
      }
    )

    bmci9 = Document(
        page_content="'hh:mm', 'CARTES', 'bmsh', 'ëdgJl ëJbw', 'Lzjsll sLyjI', 'Bonque Mouritonienne pour le Commerce Internotional', 'PRODUIT CARTE VISA', 'ELECTRON', '444003XXXXXX6512', 'EXPIRE', 'a', '1n9i0096', 'quxll +ine', 'Ququllollo lzallo', 'STATUT DE CARTE :', 'Activé', 'OPÉRATIONS SUR CARTE:', 'Commander une carte', 'Historique des demandes', 'Opposition', 'sur carte'",
        metadata= {
            "type": "Services - Cartes",
            "bank": "BMCI Mobile",
            "response": "Il s'ajit d'une capture d'ecran de la page dedier pour les services des cartes BMCI depuis l'application BMCI Mobile, elle est accessible depuis la page d'acceuil -> Service (en bas de l'ecran) -> Cartes -> et cliquer sur les 3 traits à droite. Elle presente les cartes de l'utilisateur, et les derniers mouvements effectues sur les cartes. Le lient peux commander une carte: les cartes proposees sont Carte VISA Electron, Carte VISA Gold, Carte GIMTEL et Carte VISA Prepayé. il peut également voir l'historique des demandes et faire une opposition sur une carte",
            "category" : "Support BEL"  
      }
    )

    bmci10 = Document(
        page_content="'hh:mm',  'DOCUMENTS CLIENT', 'Document', 'Compte à débiter', 'Date d'exécution', 'recherche', 'Aucun document trouvé'",
        metadata= {
            "type": "Services - Documents Client",
            "bank": "BMCI Mobile",
            "response": "Il s'ajit d'une capture d'ecran de la page dedier pour les documents des clients, elle est accessible depuis la page d'acceuil -> Service (en bas de l'ecran) -> Documents client. Le client va choisir le type de document (Avis de crédit ou Avis de débit), le compte et la date, et cliquer sur recherche",
            "category" : "Support BEL"  
      }
    )



    docs = [masrvi1,masrvi2,masrvi3,masrvi3_1, masrvi4, masrvi5, masrvi5_1, masrvi6, masrvi7, masrvi7_1, masrvi8, masrvi9,bmci1, bmci2, bmci3, bmci4, bmci5, bmci6, bmci7, bmci8, bmci9, bmci10]



    print(f"Storing data in collection {collection_name}")
    qdrant_store = Qdrant.from_documents(
        documents=documents,
        embedding=openai_embeddings,
        url=qdrant_url,
        collection_name=collection_name,
        api_key=qdrant_api_key,
    )

    for doc in docs : 
        #add last_updated with random date between now and 2 month ago
        coeff = random.randint(0, 60*60*24*30*2) # 2 months in seconds
        doc.metadata['last_updated'] = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(time.time() - coeff))
        
    # qdrant_store.add_documents(docs)
    
    store_documents_in_qdrant(
        qdrant_client=client_qdrant,
        embeddings_model=openai_embeddings,
        collection_name=collection_name,
        documents=docs + documents,
    )
    
    
    

    

# Step 5: Re-enable indexing
def set_collection_indexing(collection_name):
    client_qdrant.update_collection(
        collection_name=collection_name,
        hnsw_config=models.HnswConfigDiff(m=16), # Enable HNSW
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000), # Enable indexing 
    )


















# ---------------------------------------------------------------------------------------------
    
from qdrant_client.http.models import PointStruct
def store_documents_in_qdrant(
    qdrant_client: Any,
    embeddings_model: Any,
    collection_name: str,
    documents: List[Document],
    batch_size: int = 100
) -> None:
    """
    Stores a list of langchain_core.Document objects in Qdrant,
    mapping document metadata directly to Qdrant payload.

    Args:
        qdrant_client (Any): An initialized QdrantClient instance.
        embeddings_model (Any): An initialized embeddings model (e.g., OpenAIEmbeddings).
        collection_name (str): The name of the Qdrant collection to store documents in.
        documents (List[Document]): A list of langchain_core.Document objects to store.
        batch_size (int): The number of documents to upsert in each batch.
                          Defaults to 100 for efficiency.
    """
    if not documents:
        print("No documents provided to store.")
        return

    print(f"--- Starting to store {len(documents)} documents in Qdrant collection '{collection_name}' ---")

    points_to_upsert = []
    processed_count = 0

    try:
        # Ensure the collection exists (optional, you might have it pre-created)
        # qdrant_client.recreate_collection(
        #     collection_name=collection_name,
        #     vectors_config=vectors_config # You'll need to define this based on your embedding model's dimension
        # )
        # Or just check if exists and create if not
        # from qdrant_client.http.models import VectorParams, Distance
        # qdrant_client.recreate_collection(
        #     collection_name=collection_name,
        #     vectors_config=VectorParams(size=embeddings_model.client.embedding_ctx_length, distance=Distance.COSINE),
        # )
        # Note: The above `recreate_collection` is commented out. You should handle
        # collection creation based on your embedding model's dimension (e.g., 1536 for text-embedding-ada-002)
        # and chosen distance metric (e.g., COSINE).

        for i, doc in enumerate(documents):
            # 1. Generate Embedding
            # Langchain's embed_documents is for a list, embed_query is for single
            # For a single document content, embed_query is usually sufficient
            vector = embeddings_model.embed_query(doc.page_content)

            # 2. Prepare Payload (metadata from Document becomes Qdrant payload)
            # It's good practice to ensure payload keys are strings if not already.
            payload = {k: v for k, v in doc.metadata.items()}
            # Optionally, you might want to add page_content to payload as well for retrieval,
            # but usually, you'd retrieve based on vector similarity and then fetch payload.
            payload["page_content"] = doc.page_content # Adding page_content to payload

            # 3. Create PointStruct
            # Assign a unique ID. UUID4 is a good default.
            point_id = str(uuid.uuid4())
            points_to_upsert.append(
                PointStruct(id=point_id, vector=vector, payload=payload)
            )

            # 4. Batch Upsert
            if len(points_to_upsert) >= batch_size:
                print(f"Upserting batch of {len(points_to_upsert)} documents...")
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points_to_upsert,
                    wait=True # Wait for the operation to complete
                )
                processed_count += len(points_to_upsert)
                points_to_upsert = [] # Clear the batch list

        # Upsert any remaining documents that didn't fill a full batch
        if points_to_upsert:
            print(f"Upserting final batch of {len(points_to_upsert)} documents...")
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points_to_upsert,
                wait=True
            )
            processed_count += len(points_to_upsert)

        print(f"--- Successfully stored {processed_count} documents in collection '{collection_name}' ---")
    
        report = {
                "totalUseCases": processed_count,
                "status": "success"
            }
        
        return report

    except Exception as e:
        print(f"An error occurred during document storage: {e}")
        # You might want to log the error or re-raise it depending on your error handling strategy.



def store_use_cases_in_qdrant(file_path:str = None,):
    # Load Excel data
    df = pd.read_excel(file_path, sheet_name='use_cases')
    
    documents = []
    for _, row in df.iterrows():
        # Create dense text for embedding
        content = f"""
        INTENT: {row['intent']}
        CATEGORY: {row['categorie']}
        DESCRIPTION: {row['description']}
        EXAMPLES: {row['exemples_utilisateur']}
        STRATEGY: {row['strategie_reponse']}
        ESCALATION: {row['escalade']}
        """
        
        # Store all metadata
        payload = {
            "intent": row["intent"],
            "category": row["categorie"],
            "strategy": row["strategie_reponse"],
            "escalation": row["escalade"],
            "examples": row["exemples_utilisateur"],
            "remarques": row["remarques"]
        }
        
        documents.append(Document(page_content=content, payload=payload))
    
    # Store in dedicated collection
    return store_documents_in_qdrant(
            qdrant_client=client_qdrant,
            embeddings_model=openai_embeddings,
            collection_name="clients",
            documents=documents
        )
